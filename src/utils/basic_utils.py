import numpy as np
import random
import logging, logging.handlers
import coloredlogs
import torch
import time 

def get_logger(name, log_file_path=None, fmt="%(asctime)s %(name)s: %(message)s",
               print_lev=logging.DEBUG, write_lev=logging.INFO):
    logger = logging.getLogger(name)
    # Add file handler
    if log_file_path:
        formatter = logging.Formatter(fmt)
        file_handler = logging.handlers.RotatingFileHandler(log_file_path)
        file_handler.setLevel(write_lev)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # Add stream handler
    coloredlogs.install(level=print_lev, logger=logger,
                        fmt="%(asctime)s %(name)s %(message)s")
    return logger


def count_parameters(model):
    train_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        train_params += parameter.numel()
    print(f"Total Trainable Params: {train_params}")



def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def  compute_tiou(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return float(intersection) / (union + 1e-9)


def compute_overlap(pred, gt):
    # check format
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    pred = pred if pred_is_list else [pred]
    gt = gt if gt_is_list else [gt]  #将其转换为列表中的列表，才能够使后面找每一对左边界最大值变为广播操作
    # compute overlap
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(1e-12, union_right - union_left)
    overlap = 1.0 * inter / union
    # reformat output
    overlap = overlap if gt_is_list else overlap[:, 0]
    overlap = overlap if pred_is_list else overlap[0]  #输出得到每一对的重叠比例
    return overlap


def time_to_index(start_time, end_time, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) / float(num_units) * duration
    e_times = np.arange(1, num_units + 1).astype(np.float32) / float(num_units) * duration
    candidates = np.stack([np.repeat(s_times[:, None], repeats=num_units, axis=1),
                           np.repeat(e_times[None, :], repeats=num_units, axis=0)], axis=2).reshape((-1, 2))
    overlaps = compute_overlap(candidates.tolist(), [start_time, end_time]).reshape(num_units, num_units)
    start_index = np.argmax(overlaps) // num_units
    end_index = np.argmax(overlaps) % num_units
    return start_index, end_index, overlaps


def index_to_time(start_index, end_index, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) * duration / float(num_units)
    e_times = np.arange(1, num_units + 1).astype(np.float32) * duration / float(num_units)
    start_time = s_times[start_index]
    end_time = e_times[end_index]
    return start_time, end_time


def fetch_feats_by_index(ori_feats, indices):
    B, L = indices.shape
    filtered_feats = ori_feats[torch.arange(B)[:, None], indices]
    return filtered_feats

def compute_tiou_vectorized(pred, gt):
    """
    计算所有候选的 IoU 值
    """
    pred_start, pred_end = pred[:, 0], pred[:, 1]
    gt_start, gt_end = gt[0], gt[1]

    inter_start = torch.max(pred_start, gt_start)
    inter_end = torch.min(pred_end, gt_end)
    inter_len = torch.clamp(inter_end - inter_start, min=0)

    union_len = (pred_end - pred_start) + (gt_end - gt_start) - inter_len
    iou = inter_len / union_len
    return iou

class Evaluator(object):

    def __init__(self, tiou_threshold=[0.1, 0.3, 0.5], topks=[1, 5, 10, 50, 100]):
        self.tiou_threshold = tiou_threshold
        self.topks = topks

    def eval_instance(self, pred, gt, topk):
        """_summary_

        Args:
            pred (_tensor_): _description_:(100,2)表示每个query的100个候选结果(开始和结束时刻)
            gt (_tensor_): _description_ :(2) 表示每个query的真实结果
            topk (_单值_): _description_：表示从100个候选中选择前topk个

        Returns:
            _type_: _description_
        """
        correct = {str(tiou):0 for tiou in self.tiou_threshold}
        find = {str(tiou):False for tiou in self.tiou_threshold}
        #{'0.1': 0, '0.2': 0, '0.3': 0}, {'0.1': False, '0.2': False, '0.3': False}
        if len(pred) == 0:
            return correct

        if len(pred) > topk:
            pred = pred[:topk]
        
        # 计算所有候选的 tiou 值,从tensor转换到numpy会将数据转换为cpu上
        ious = compute_tiou_vectorized(pred, gt)
        best_tiou = torch.max(ious).item()
            
        for tiou in self.tiou_threshold:
            tiou_str = str(tiou)
            mask = (ious >= tiou) & (~find[tiou_str])  # 直接在 tensor 上操作
            correct[tiou_str] = int(torch.any(mask))  # 使用 torch.any(mask) 判断
            find[tiou_str] = torch.any(mask)  # 更新 find 字典
        
        return correct, best_tiou

    def eval(self, preds, gts):
        """ Compute R@1 and R@5 at predefined tiou threshold [0.3,0.5,0.7]
        Args:
            preds: list；元素个数为query的总数，没有了video的概念 num_all_querys,100,2
            gts: list；元素个数为query的总数，没有了video的概念 num_all_querys,2
        Return:
            correct: flag of correct at predefined tiou threshold [0.3,0.5,0.7]
        """
        print("preds,gts:{},{}".format(len(preds),len(gts)))    #CPU/GPU: 72044，72044
        print("type of preds,gts:{},{}".format(type(preds),type(gts)))
        eval_metric_st=time.time()
        num_instances = float(len(preds)) #应该计算的是所有的query
        print("num_instances: ",num_instances)
        miou = 0
        all_rank = dict()
        for tiou in self.tiou_threshold:
            for topk in self.topks:
                #top-k和tiou是分别计算的
                all_rank["R{}-{}".format(topk, tiou)] = 0
        
        #每个元素表示一个视频数据，用列表而不是张量的形式是因为每个视频的query数量不一样
        #在列表中可以做到不同长度的存储，而张量不行，但是这里的评价标准是视频为单位还是query？
        count=0
        #对于每一个query去单独计算，每一行计算，preds,gts 72044,100,2 ; 72044,2
        count_st=time.time()
        for pred,gt in zip(preds, gts): 
            #每次拿出一个query进行计算
            for topk in self.topks:
                #在eval_instance中计算得到的best_iou并没有参与到后续计算中
                correct, iou = self.eval_instance(pred, gt, topk=topk)  #因为内部是一个一个计算所以时间开销特别大？
                for tiou in self.tiou_threshold:
                    all_rank["R{}-{}".format(topk, tiou)] += correct[str(tiou)]
                    
            # count+=1
            # if count%1000==0:
            #     count_ed=time.time()
            #     print("count:{} ,time:{} ".format(count,count_ed-count_st))
            #     count_st=time.time()
                
                # miou += iou
        #这个指标不是按照每个视频来计算，而是按照query来计算，有点奇怪
        print("ending eval compute")
        for tiou in self.tiou_threshold: 
            for topk in self.topks:
                all_rank["R{}-{}".format(topk, tiou)] /= num_instances

        # miou /= float(num_instances)
        
        eval_metric_et=time.time()
        print("eval_metric time: ",eval_metric_et-eval_metric_st)
        return all_rank, miou