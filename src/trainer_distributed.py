import os
import json
import time
from tqdm import tqdm
import torch
from torch.optim import AdamW, lr_scheduler

from .models import SOONet
from .utils import Evaluator, get_logger
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np

def cleanup():
        dist.destroy_process_group()
        
class Trainer(object):
    def __init__(self, mode, save_or_load_path, cfg,rank):
        # self.device = torch.device(cfg.device_id) if torch.cuda.is_available() else torch.device("cpu")
        self.model = SOONet(cfg)
        self.device = torch.device(f'cuda:{rank}')
        self.rank=rank
        #将模型中的所有 BatchNorm 层转换为 SyncBatchNorm 层。SyncBatchNorm（同步批归一化）
        #是一种在分布式训练中使用的批归一化层，它可以在多 GPU 环境中同步计算统计量，从而提高模型的稳定性和性能。
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model) 
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Total parameters: {total_params}')
        # print(rank)
        self.model = self.model.to(rank)
        #最后这个参数会检查模型反向传播有哪些参数没有被使用，在一定程度上会增加时间开销
        self.model = DDP(self.model, device_ids=[rank],find_unused_parameters=True)
         
        self.evaluator = Evaluator(tiou_threshold=cfg.TEST.EVAL_TIOUS, topks=cfg.TEST.EVAL_TOPKS)
        self.save_or_load_path = save_or_load_path
        log_dir = os.path.join(save_or_load_path, "log")
        ckpt_dir = os.path.join(save_or_load_path, "ckpt")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.cfg = cfg

        if mode == "train":
            with open(os.path.join(log_dir, "config.json"), 'w') as f:
                js = json.dumps(cfg, indent=2)
                f.write(js)

            self.optimizer = self.build_optimizer(cfg)
            self.scheduler = lr_scheduler.StepLR(self.optimizer, cfg.OPTIMIZER.LR_DECAY_STEP, 
                                    gamma=cfg.OPTIMIZER.LR_DECAY, last_epoch=-1, verbose=False)
        
    
    def build_optimizer(self, cfg):
        no_decay = ['bias', 'layer_norm', 'LayerNorm']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': cfg.OPTIMIZER.WD},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.OPTIMIZER.LR)
        
        # # 打印优化器中包含的参数名
        # for i, param_group in enumerate(optimizer.param_groups):
        #     print(f"Parameter Group {i}:")
        #     for n, p in self.model.named_parameters():
        #         if p in param_group['params']:
        #             print(f"  {n}")
        
        return optimizer


    def train(self, train_loader, test_loader,train_sampler):
        logger = get_logger("TRAIN", log_file_path=os.path.join(self.log_dir, "train.log"))
        # self.model.to(self.device)
        self.train_epoch(0, train_loader, test_loader, logger, self.cfg,train_sampler)
        
        # self.train_fake(0, train_loader, test_loader, logger, self.cfg,train_sampler)


    def eval(self, test_loader):
        logger = get_logger("EVAL", log_file_path=os.path.join(self.log_dir, "eval.log"))

        resume_path = os.path.join(self.ckpt_dir, "best.pth")
        logger.info("Load trained model from: {}".format(resume_path))

        saver_dict = torch.load(resume_path, map_location="cpu")
        state_dict = saver_dict["model"]
        self.model.load_state_dict(state_dict, strict=True)
        # self.model.to(self.device)
        self.model.eval()
        logger.info("Load trained model succeed.")

        all_rank, miou = self.eval_epoch(test_loader)
        for k, v in all_rank.items():
            logger.info("{}: {:.4f}".format(k, v))


    def test(self, test_loader):
        logger = get_logger("TEST", log_file_path=os.path.join(self.log_dir, "test.log"))

        resume_path = os.path.join(self.ckpt_dir, "best.pth")
        logger.info("Load trained model from: {}".format(resume_path))

        saver_dict = torch.load(resume_path, map_location="cpu")
        state_dict = saver_dict["model"]
        self.model.load_state_dict(state_dict, strict=True)
        # self.model.to(self.device)
        self.model.eval()
        logger.info("Load trained model succeed.")

        start = time.time()
        test_instances = []
        with torch.no_grad():
            for batch in tqdm(test_loader, total=len(test_loader)):
                scores, bboxes = self.model(
                    # query_feats=batch["query_feats"].to(self.device), 
                    # query_masks=batch["query_masks"].to(self.device), 
                    # video_feats=batch["video_feats"].to(self.device),
                    # start_ts=batch["starts"].to(self.device),
                    # end_ts=batch["ends"].to(self.device),
                    # scale_boundaries=batch["scale_boundaries"].to(self.device),
                    
                    query_feats=batch["query_feats"].to(self.rank), 
                    query_masks=batch["query_masks"].to(self.rank), 
                    video_feats=batch["video_feats"].to(self.rank),
                    start_ts=batch["starts"].to(self.rank),
                    end_ts=batch["ends"].to(self.rank),
                    scale_boundaries=batch["scale_boundaries"].to(self.rank),
                    
                )
                for i in range(len(bboxes)):
                    instance = {
                        "vid": batch["vid"],
                        "duration": batch["duration"],
                        "qid": batch["qids"][0][i],
                        "text": batch["texts"][0][i],
                        "timestamp": batch["timestamps"][i].numpy().tolist(), 
                        "pred_scores": scores[i],
                        "pred_bboxes": bboxes[i]
                    }
                    test_instances.append(instance)

        logger.info("cost time: {}".format(time.time() - start))
        result_path = os.path.join(self.log_dir, "infer_result.json")
        with open(result_path, 'w') as f:
            res = json.dumps(test_instances, indent=2)
            f.write(res)

    
    def train_epoch(self, epoch, train_loader, test_loader, logger, cfg,train_sampler):
        #在train_epoch的过程中就会去进行eval
        self.model.train()
        best_r1 = 0
        batch_st=time.time()
        train_sampler.set_epoch(epoch)
        for j in range(40):  #j轮迭代，在dataset中设置的num_epoch参数为1，并没有使用多轮的数据
            epoch_hitR=0
            classification_loss=0
            for i, batch in enumerate(tqdm(train_loader, total=len(train_loader), desc="Training")):
                #train_loader是从dataset中的self.sample拿数据,sample数据本身就是包含了epoch信息，已经将每个视频重复了epoch次
                #因此从train_loader中拿的数据就是epoch次的数据，不需要再重复epoch次
                # print("rank,i,video_feats shape:",self.rank,i,batch["video_feats"].shape)
                batch_et = time.time()
                training_st=time.time()
                loss_dict,hit_ratio = self.model(  #hit_ratio是一个视频七条样本的均值
                    query_feats=batch["query_feats"].to(self.rank), 
                    video_feats=batch["video_feats"].to(self.rank),
                    video_events=batch["video_events"].to(self.rank),
                    timestamps=batch["timestamps"].to(self.rank),
                    event_feats=batch["event_feats"][0], #其实并没有用到
                )
                epoch_hitR+=hit_ratio
                        

                total_loss = loss_dict["total_loss"]  #训练一个小批次之后就更新参数权重(7个query)
                self.optimizer.zero_grad()
                total_loss.backward()
                classification_loss+=loss_dict["classficate_loss"]
                
                
                #事实上在编码时，并没有使用到这个模块，所以打印出来的梯度是None
                # print(self.model.module.abstract_encoder.linear.bias.grad[:10])  
                # print(self.model.module.abstract_encoder.linear.bias.requires_grad)
                # print(self.model.module.regressor.fc_q.bias.grad[:10])
                # print(self.model.module.regressor.fc_q.bias.requires_grad)
                self.optimizer.step()
                self.scheduler.step()
                
                training_et=time.time()

                #在记录实验结果时，也是主卡记录就可以，下面这种写法，使用了几张卡，则会记录几次，应该加上判断rank==0
                # if i % cfg.TRAIN.LOG_STEP == 0 and self.rank==0:  #LOG_STEP设置值为200，表示读了200个数据，但是并不能表示对所有数据进行了一轮迭代吧?
                if i==len(train_loader)-1 and self.rank==0:  #LOG_STEP设置值为200，表示读了200个数据，但是并不能表示对所有数据进行了一轮迭代吧?
                    log_str = f"Step: {i}, "
                    for k, v in loss_dict.items():
                        log_str += "{}: {:.3f}, ".format(k, v)
                    logger.info(log_str[:-2])
                #上面200步，是保存训练的损失，下面2000步是保存验证集的性能
                if i > 0 and i % cfg.TRAIN.EVAL_STEP == 0:  #2000个step的时候保存一次模型
                # if i > 0 and i == len(train_loader)-1:  #训练完一轮训练数据之后
                    #每次进入eval_epoch都会重新启动num_workers，因此和train_loader一样，在第一个batch时时间开销比较大，但是后面的开销不大
                    #miou在计算中就是0
                    # all_rank, miou,Test_hit_rate = self.eval_epoch(test_loader)  
                    data,Test_hit_rate = self.eval_epoch(test_loader)  
                    all_rank, miou=data
                    
                    write_logfile_st=time.time()
                    if self.rank==0: #只有在主卡上记录日志
                        logger.info("step: {}".format(i))
                        for k, v in all_rank.items():
                            logger.info("{}: {:.4f}".format(k, v))

                        r1 = all_rank["R1-0.5"]
                        if r1 > best_r1:
                            best_r1 =r1
                            saver_dict = {
                                "step": i,
                                "r1-0.5": r1,
                                "model": self.model.module.state_dict(),
                                "optimizer": self.optimizer.state_dict()
                            }
                            save_path = os.path.join(self.ckpt_dir, "best.pth")
                            torch.save(saver_dict, save_path)
                    
                    write_logfile_et=time.time()
                    # print(f"write logfile time: {write_logfile_et-write_logfile_st}")
                    self.model.train()
                    print("test hit rate: ",Test_hit_rate)
                #打印每次的数据加载时间和训练时间
                training_time = training_et - training_st
                data_loading_time = batch_et - batch_st
                
                # print(f"Step {i + 1}: Data loading time: {data_loading_time:.4f} seconds, Training time: {training_time:.4f} seconds")
                batch_st = time.time()
            logger.info("best R1-0.5: {:.4f}".format(best_r1))
            print(f"epoch {j} Training hit ratio:{epoch_hitR/len(train_loader)},classification loss:{classification_loss/len(train_loader)}")
            print(f"epoch {j} Test hit ratio:{Test_hit_rate}")
        
        
    def eval_epoch(self, test_loader):
        self.model.eval()  #将self.training改为false操作
        #eval的时候和training执行的操作不一样，可能导致时间变长
        preds, gts = list(), list()
        with torch.no_grad():
            eval_batch_st = time.time()
            all_hit_ratio=0
            #从dataset中拿出的一条数据是一个视频和其对应的所有query,但在模型forward时，还是使用yaml文件中的batch去计算的
            for i, batch in enumerate(test_loader):
                eval_batch_et = time.time()
                eval_model_st=time.time()
                #在验证的时候模型中使用top 100,得到的数据都在CPU上
                #在测试时，目的是为了得到预测出来区间戳
                bboxes,batch_hit_ratio = self.model(  #里面计算出来的hit_ratio是一个视频下的所有query的hit_ratio，外面计算的是所有视频的hit_Ratio
                    # query_feats=batch["query_feats"].to(self.rank), 
                    # query_masks=batch["query_masks"].to(self.rank), 
                    # video_feats=batch["video_feats"].to(self.rank),
                    # start_ts=batch["starts"].to(self.rank),
                    # end_ts=batch["ends"].to(self.rank),
                    # scale_boundaries=batch["scale_boundaries"].to(self.rank),
                    
                    query_feats=batch["query_feats"].to(self.rank), 
                    video_feats=batch["video_feats"].to(self.rank),
                    video_events=batch["video_events"].to(self.rank),
                    timestamps=batch["timestamps"].to(self.rank),
                )
                #bboxes是一个视频下的的all_query,topk,2
                #每个视频extend后,all_videos,all_query,topk,2【元素个数为all_videos】
                #也可以进行numpy或者张量化，因为每个视频下query个数不同，但是可以累加，最终表示所有视频query的总和
                preds.extend(bboxes)  #extend不是append
                # # print("preds shape:",np.array(preds).shape)  #6,100
                # #timestamps是一个query一个，但此处是视频单位，因此会有多个,得到个视频一批的query开始和结束时间
                gts.extend([i for i in batch["timestamps"]])
                
                all_hit_ratio+=batch_hit_ratio
                
                
                # eval_model_et=time.time()
                # print(f"eval batch time: {eval_batch_et-eval_batch_st}, eval model time: {eval_model_et-eval_model_st}")
                # eval_batch_st = time.time()
        
        #得到所有批次的数据之后统一计算性能指标
        #并不应该选这么多个事件出来，还要再进一步控制一下，但标准是什么？？
        #每个查询得到的事件不一定一样，有[1785,2]也有[1651,2]，但这个表示的是什么呢？好像是所有的查询直接堆积起来前xx个元素表示的是第一个视频的所有查询
        preds=torch.stack(preds,dim=0)
        gts=torch.stack(gts,dim=0)
        # print("2000 Test hit ratio: ",all_hit_ratio/len(test_loader))
        # return 0,all_hit_ratio/len(test_loader)  #所有视频的查询命中
        return self.evaluator.eval(preds, gts),all_hit_ratio/len(test_loader)