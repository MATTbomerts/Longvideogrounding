import argparse
import torch.utils
import yaml
from easydict import EasyDict as edict
import torch
import torch.utils.data as data
import os

# from src.mad_testDataset import MADDataset

from .datasets import *

from .trainer_distributed import Trainer
from .utils import set_seed
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import pdb
from torch.utils.data.distributed import DistributedSampler
from datetime import timedelta
import time
import datetime
from tqdm import tqdm
import tracemalloc
import numpy as np

os.environ['NCCL_DEBUG'] = 'TRACE'  #添加此行代码，显示更详细的报错信息


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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345' #不能这个端口正在被使用，必须是空闲的状态
    os.environ['LOCAL_RANK'] = str(rank) #单机多卡上rank和local_rank是一样的
    torch.cuda.set_device(rank)
    print(f"Process {rank} - init process group")
    # dist.init_process_group("nccl", rank=rank,world_size=world_size, timeout=timedelta(seconds=5))
    dist.init_process_group("nccl", rank=rank, init_method='env://', world_size=world_size, timeout=datetime.timedelta(seconds=60))
    print(f"Process {rank} - init process group done")

def cleanup():
    dist.destroy_process_group()

# def main(rank, world_size):
def main():
    parser = argparse.ArgumentParser("Setting for training SOONet Models")
    parser.add_argument("--exp_path", type=str)
    parser.add_argument("--config_name", type=str)
    parser.add_argument('--rank', type=int) #rank是跨所有节点(机器)的全局标识符，而local_rank是每个节点(机器)上的本地标识符
    parser.add_argument('--local_rank', type=int)  #在单机多卡的情况下，local_rank和rank可以互换使用
    # parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--mode", type=str, default="train")

    opt = parser.parse_args()
    rank=opt.local_rank
    world_size=torch.cuda.device_count()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'  #这个可用的设备数，需要和world_size一致
    setup(rank, world_size)
    
    assert rank < torch.cuda.device_count(), f"Rank {rank} exceeds available device count {torch.cuda.device_count()}"
    
    config_path = "conf/{}.yaml".format(opt.config_name)
    with open(config_path, 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    # cfg.device_id = opt.device_id
    # torch.cuda.set_device(opt.device_id)
    set_seed(cfg.SEED)
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    dset = cfg.DATASET
    anchor_St_time=time.time()
    if dset.lower() == "mad":
        #pre_load配置文件中为false

        
        trainset = MADDataset("train", cfg, pre_load=cfg.DATA.PRE_LOAD) if opt.mode == "train" else list()
        testset = MADDataset("test", cfg, pre_load=cfg.DATA.PRE_LOAD)

        
    else:
        raise NotImplementedError
    if rank==0:
        print("Train batch num: {}, Test batch num: {}".format(len(trainset), len(testset)))
        print(cfg)
        
    
    if opt.mode == "train":
        
        #因为本身的Sampler实现方式
        train_sampler=DistributedSampler(trainset, num_replicas=world_size, rank=rank)
        # train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank)

        train_loader = data.DataLoader(trainset, 
                                batch_size=1,  #论文中说 1 video with 32 queries
                                num_workers=cfg.TRAIN.WORKERS,
                                #使用分布式训练，dataloader中的shuffle应该设置为false,因为DistributedSampler会打乱数据
                                shuffle=False,  
                                #使用分布式训练需要加入sampler参数
                                sampler=train_sampler,
                                collate_fn=trainset.collate_fn,
                                drop_last=False
                            )

    test_loader = data.DataLoader(testset, 
                            batch_size=1,
                            num_workers=cfg.TEST.WORKERS,
                            shuffle=False,
                            collate_fn=testset.collate_fn,
                            drop_last=False
                        )
    
    tracemalloc.start()
    vid_event_dict=dict()
    for i, batch in enumerate(tqdm(train_loader, total=len(train_loader), desc="Training")):
        # batch是从train_loader中获取的一批数据
        # 在这里进行数据处理和模型训练
        # vid=batch["vid"]
        
        # if len(vid)!=1:
        #     assert "Only support len size 1"
        #     exit(0)
        query_feats=batch["query_feats"]
        video_feats=batch["video_feats"].numpy()
        # print(video_feats.shape)
        # if vid[0] not in vid_event_dict:  #不会重复计算
        #     event=region_growing_event_clustering(video_feats[0],0.75)
        #     vid_event_dict[vid[0]]=event
        # else:  #事实证明有效
        #     print("already computed!")
    
    anchor_Et_time=time.time()
    current, peak = tracemalloc.get_traced_memory()
    # 停止跟踪内存分配
    tracemalloc.stop()
    print("anchor time spent: ",anchor_Et_time-anchor_St_time)
    print(f"Current memory usage: {current / 10**6} MB")
    print(f"Peak memory usage: {peak / 10**6} MB")
    print(len(vid_event_dict))
    
    cleanup()
    

def run_training(world_size):
    # mp.spawn(main,
    #          args=(world_size,),
    #          nprocs=world_size,
    #          join=True)
    
    main(ot.local_rank,world_size)



if __name__ == "__main__":
    
    world_size = torch.cuda.device_count()
    main()
    # run_training(world_size)

# CUDA_VISIBLE_DEVICES=6  NCCL_P2P_LEVEL=PIX  python -m  torch.distributed.launch --nproc_per_node=1  --nnodes=1   --exp_path test_out --config_name soonet_mad --mode train  main_distributed.py

#CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=PIX python -m torch.distributed.launch --nproc_per_node=1  --nnodes=1  --master_port=12345 -m src.main_testTime --exp_path test_out --config_name soonet_mad --mode train