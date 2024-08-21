import numpy as np
import random
import torch.nn.functional as F
import logging, logging.handlers
import coloredlogs
import torch
import time 
from collections import Counter
import torch.nn as nn

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

    inter_start = torch.max(pred_start, gt_start)  #广播机制并行化计算
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
        ious = compute_tiou_vectorized(pred, gt)  #这部分是批量化计算的
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
        # print("preds,gts:{},{}".format(len(preds),len(gts)))    #CPU/GPU: 72044，72044
        # print("type of preds,gts:{},{}".format(type(preds),type(gts)))
        eval_metric_st=time.time()
        num_instances = float(len(preds)) #应该计算的是所有的query
        
        # print("num_instances: ",num_instances)
        miou = 0
        all_rank = dict()
        for tiou in self.tiou_threshold:  #但其实数量不多，直接双重for循环问题也不大
            for topk in self.topks:
                #top-k和tiou是分别计算的
                all_rank["R{}-{}".format(topk, tiou)] = 0
        
        #每个元素表示一个视频数据，用列表而不是张量的形式是因为每个视频的query数量不一样
        #在列表中可以做到不同长度的存储，而张量不行，但是这里的评价标准是视频为单位还是query？
        count=0
        #对于每一个query去单独计算，每一行计算，preds,gts 72044,100,2 ; 72044,2
        count_st=time.time()  #这一部分时间计算开销十分大
        for pred,gt in zip(preds, gts):   #如果直接使用preds和gts的话，会导致内存占用过大，因为preds和gts是所有的query的结果
            #每次拿出一个query进行计算
            for topk in self.topks:
                #在eval_instance中计算得到的best_iou并没有参与到后续计算中
                correct, iou = self.eval_instance(pred, gt, topk=topk)  #因为内部是一个一个计算所以时间开销特别大？
                for tiou in self.tiou_threshold:
                    all_rank["R{}-{}".format(topk, tiou)] += correct[str(tiou)]
                    

        for tiou in self.tiou_threshold: 
            for topk in self.topks:
                all_rank["R{}-{}".format(topk, tiou)] /= num_instances
        
        eval_metric_et=time.time()
        print("eval_metric time: ",eval_metric_et-eval_metric_st)
        return all_rank, miou
    

# 计算两帧特征之间的相似度（余弦相似度）
def calculate_similarity(feature1, feature2):
    dot_product = np.dot(feature1, feature2)
    norm_a = np.linalg.norm(feature1)  #L2范数，表示向量的长度
    norm_b = np.linalg.norm(feature2)
    similarity = dot_product / (norm_a * norm_b) #向量积除以向量长度的乘积等于相似度
    return similarity

# 根据相似度进行区域生长
def region_growing_event_clustering(features, similarity_threshold):
    num_frames = len(features)
    # 初始化事件标签为-1,一开始所有的帧都标记为没有归类所属事件，大小和帧数一样
    events = np.zeros(num_frames, dtype=int) - 1  
    current_event = 0  #事件的记号，依次递增表示第几个事件
    
    for i in range(num_frames):
        #如果当前帧没有标记为任何事件，防止当前帧在聚集事件之后将相邻帧也进行了标记，后续还要对相邻帧进行计算的重复开销
        if events[i] == -1:  #第一帧处理是没有事件标记的
            events[i] = current_event
            queue = [i]
            while queue:
                current_frame = queue.pop(0)
                #neighbor也只是帧索引序号，当前帧的左右相邻两帧，左右两帧进行遍历计算
                for neighbor in [current_frame - 1, current_frame + 1]:
                    #相邻帧在合理的范围内，且没有被标记为任何事件
                    if 0 <= neighbor < num_frames and events[neighbor] == -1:
                        similarity = calculate_similarity(features[current_frame], features[neighbor])
                        if similarity > similarity_threshold:
                            events[neighbor] = current_event
                            queue.append(neighbor)
            current_event += 1
    
    return events


def events_modify(events):
    """_summary_:将事件中出现次数低于5帧的事件进行相邻事件的合并，以避免因抖动、遮挡导致的事件划分不准确
    先统计出所有的长度低于5帧的事件(不足1s)，然后遍历这些事件，从当前事件向左找第一个长度大于5帧的事件：三种情况
    1：左边事件长度大于5帧，直接将当前事件的所有帧标记为左边事件
    2：左边事件长度小于5帧，因为是for循环遍历的方式，本身事件就是从小到大的，因此遍历到的第一个事件就是最第一个长度小于5帧的事件，不存在此情况
    3：左边没有事件，当前事件就是第一个事件，直接和1号事件进行合并（但测试集中0号事件长度都长于5帧）
    todo: 有可能0号事件和1号事件加起来也不足5帧
    Args:
        events (_type_): 事件索引，从0开始顺序递增，没有间隔

    Returns:
        _type_: 融合后的事件索引，每个事件长度大于5帧，但不保证编号连续
    """
    #统计事件中出现次数低于5的事件个数
    #统计每个元素的出现次数
    counts = Counter(events)
    require_count=5
    # 找出出现次数不超过5次的元素
    elements_with_few_occurrences = [elem for elem, count in counts.items() if count <require_count ]
    for element in elements_with_few_occurrences:  #是从所有小于5帧的元素中遍历
        # print(element)
        if counts[element]>=require_count:  #如果当前事件的长度已经大于5帧，就不需要再合并了,因为后面的操作中会改变（0号事件就不足的情况）
            continue
        events_start=np.where(events==element)[0][0]
        events_end=np.where(events==element)[0][-1]
        left_event=element+1
        for elem in range(element-1,-1,-1):
            if counts[elem]>=require_count: #也有可能一开始第0个事件就不满足5帧
                left_event=elem
                break
        if left_event==element+1: # 如果左边没有找到满足条件的事件
            left_event=element-1 #解决一开始第0个事件就不满足5帧的情况
            
        if element==0:  #第0个事件就长度不足5，但是还是可能第0个和第一个事件之和也不足5帧
            left_event=element+1 #直接替换为1号事件,因为得到的事件本身是连续的
        
        select_event=left_event  
        # 直接将events里面的元素值进行修改，后面计算IOU的时候就直接用即可
        events[events_start:events_end+1]=len(events[events_start:events_end+1])*[select_event]
        counts[select_event]+=counts[element]
        del counts[element] #删除当前元素，del 之后，该元素的值就变为0了，因此也不会在向左找的时候找到它
    return events  #一开始将return放在了for循环里面，导致只返回了最后一个元素的修改结果


class AttentionModule(torch.nn.Module):
    def __init__(self, num_queries, input_dim):
        super(AttentionModule, self).__init__()
        self.num_queries = num_queries
        self.input_dim = input_dim
        # 定义查询向量
        self.queries = torch.nn.Parameter(torch.randn(num_queries, input_dim))
        # 线性层，将帧特征映射到注意力分数
        self.linear = torch.nn.Linear(input_dim, input_dim)
        
    def forward(self, event_unit):
        # event_unit 的维度应为 (num_frames, input_dim)
        num_frames = event_unit.size(0)
        
        event_unit=self.linear(event_unit)
        # 计算查询向量与帧特征的注意力分数
        # queries(num_queries,hidden dim) event_unit(num_frames,hidden dim) -> (num_queries,num_frames)
        query_scores = torch.matmul(self.queries, event_unit.T) 
        
        # 计算注意力权重
        # 每一行上进行 softmax，得到每个查询对每帧的注意力权重(合理)
        weights = F.softmax(query_scores, dim=1) # 在每个查询向量的维度上进行 softmax
        
        # 加权平均每帧的特征，得到每个查询的抽象特征
        #weights(num_queries,num_frames) event_unit(num_frames,hidden dim) -> (num_queries,hidden
        attended_features = torch.matmul(weights, event_unit)  # (num_queries, input_dim)
        
        # 返回结果
        return attended_features
    
class EventQueryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(EventQueryClassifier, self).__init__()
        # 定义一个前馈神经网络
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, query_feats, event_feats):
        # 含义没有问题
        # 扩展维度以便进行广播
        # 查询在第二个维度扩展事件，表示每个查询对应每个事件复制一次
        query_feats_expanded = query_feats.unsqueeze(1).expand(-1, event_feats.size(0), -1)
        # 事件在第一个维度扩展查询，表示每个事件对应每个查询复制一次
        event_feats_expanded = event_feats.unsqueeze(0).expand(query_feats.size(0), -1, -1)

        # 最终的结果就相当于是 每个查询和每个事件的组合(num_query,num_event)
        # 拼接查询特征和事件特征 （查询、事件，特征维度*2）
        combined_feats = torch.cat((query_feats_expanded, event_feats_expanded), dim=-1)

        # 将特征输入到分类器中
        x = F.relu(self.fc1(combined_feats))
        output = torch.sigmoid(self.fc2(x)).squeeze(-1)  # 输出命中概率 (k, n)
        return output
    

def get_targets(pred_timestamps,hit_indices,classifier_target,frame_timestamps,inverse_indices):
    """_summary_

    Args:
        pred_timestamps (_type_): 针对事件预测出来的时间区间
        hit_indices (_type_): 选择出来的事件
        classifier_target (_type_): 命中损失
        frame_timestamps (_type_): 查询标注的时间区间
        inverse_indices (_type_): 视频帧的事件索引

    Returns:
        _type_: _description_
    """
    hit_ratio=0
    event_overlaps_pred=list()
    for query_index,frame_timestamp in enumerate(frame_timestamps): 
        overlap_events=torch.unique(inverse_indices[frame_timestamp[0]:frame_timestamp[1]+1]) #这个overlap_events是0开始的
        classifier_target[query_index][overlap_events]=1  #针对每个查询的命中程度
        overlap_event_predict=[] #针对每个查询都得到的一个目标损失预测
        flag=0  #如果该query一个事
        for overlap_event in overlap_events: 
            #hit_indices和overlap_events都是绝对的事件编号 
            if overlap_event in hit_indices:  #overlap_event是一个一个进行遍历计算的
                #选择得到的事件、预测结果的索引和hit_indices存在压缩对应的关系，位序编号保持一致
                #原本在hit_indices中第1个元素为6(表示第6个事件)，在pred_timestamps中将会变为第一个元素值，对应的是所有事件的第6个
                #因此应该找hit_indices中值(事件)的索引，在选择后的事件中使用该索引得到该事件的预测值
                overlap_index = (hit_indices == overlap_event).nonzero(as_tuple=True)[0]
                overlap_event_predict.append(pred_timestamps[query_index][overlap_index].squeeze())#得到的是绝对的时间，如果不使用squeeze会多一个维度
                flag=1 #只要有一个命中就可以
            else: #这里这个逻辑还是比较奇怪，如果预测命中事件没有命中的话
                overlap_event_predict.append(torch.sigmoid(pred_timestamps[query_index][0].squeeze()))   #额外设置的这个值没有梯度无法更新
        if flag==1:
            hit_ratio+=1  #但是这个hit_ratio是共享得到的，不是每个query得到的，因此不准确，需要根据classficate_result来计算
        event_overlaps_pred.append(torch.stack(overlap_event_predict))
    return event_overlaps_pred,hit_ratio,classifier_target  
     