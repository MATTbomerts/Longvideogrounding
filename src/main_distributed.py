import argparse
import torch.utils
import yaml
from easydict import EasyDict as edict
import torch
import torch.utils.data as data
import os
import sys
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

os.environ['NCCL_DEBUG'] = 'TRACE'  #添加此行代码，显示更详细的报错信息

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
    if dset.lower() == "mad":
        #pre_load配置文件中为false
        dataset_sttime=time.time()
        
        trainset = MADDataset("train", cfg, pre_load=cfg.DATA.PRE_LOAD) if opt.mode == "train" else list()
        testset = MADDataset("test", cfg, pre_load=cfg.DATA.PRE_LOAD)
        dataset_ettime=time.time()
        
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
    trainer_sttime=time.time()
    #时间开销比较小
    trainer = Trainer(mode=opt.mode, save_or_load_path=opt.exp_path, cfg=cfg,rank=rank)
    trainer_ettime=time.time()
    # pdb.set_trace()
    print("trainer time: ",trainer_ettime-trainer_sttime)
    print("dataset time: ",dataset_ettime-dataset_sttime)
    if opt.mode == "train":
        trainer.train(train_loader, test_loader,train_sampler)
    elif opt.mode == "eval":
        trainer.eval(test_loader)
    elif opt.mode == "test":
        trainer.test(test_loader)
    else:
        raise ValueError(f'The value of mode {opt.mode} is not in ["train", "eval", "test"]')
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
# CUDA_VISIBLE_DEVICES=6 NCCL_P2P_LEVEL=PIX python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --use_env main_distributed.py --exp_path test_out --config_name soonet_mad --mode train
# CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=PIX  python -m torch.distributed.launch --nproc_per_node=1  --nnodes=1  --master_port=12345 -m src.main_distributed --exp_path distribute_out --config_name soonet_mad --mode train