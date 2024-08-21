import os
import h5py
import random
import math
import numpy as np
from easydict import EasyDict as edict
from collections import defaultdict
import torch
import torch.utils.data as data
import itertools
# from ..utils import compute_overlap,region_growing_event_clustering
from ..utils.basic_utils import compute_overlap,region_growing_event_clustering,events_modify


class MADDataset(data.Dataset):
    
    def __init__(self, split, cfg, pre_load=False):
        super().__init__()
        """_summary_:
        功能：构造q2v和v2q的字典,以及总的数据样本self.samples
        q2v是一对一的关系(对应的视频，视频的总时长，对应的区间，对应的文本【不包含特征】)
        v2q是一对多的关系(一个视频和其对应的多个query id)
        samples:list结构，每个元素是一个字典，包含一个视频和batch_size个query，包含所有epoch的数据
        """
        self.split = split
        self.data_dir = cfg.DATA.DATA_DIR
        # self.snippet_length = cfg.MODEL.SNIPPET_LENGTH  #10 对应论文中C0
        # self.scale_num = cfg.MODEL.SCALE_NUM  #4  对应论文中尺度数
        # self.max_anchor_length = self.snippet_length * 2**(self.scale_num - 1)   #对应一维卷积的长度？？请见下文
        if split == "train":  #yaml文件中的两个配置在数据集初始化时使用，而不是在dataloader和训练代码中使用
            epochs = cfg.TRAIN.NUM_EPOCH
            batch_size = cfg.TRAIN.BATCH_SIZE  #7
        else:
            #测试时，bsz固定设置为100万，以便一次处理一个视频的所有query
            #在训练模式下，选择小的bsz是因为训练需要后向传播和梯度计算，需要额外的内存开销
            #验证集中每个movie下最多1500个query，最少150个query
            epochs = 1
            batch_size = 1000000
        
        self.q2v = dict()  #一对一、qid对应的样本，但不包含视频特征，只有视频id
        self.v2q = defaultdict(list) #一对多
        self.v2dur = dict()
        with open(os.path.join(self.data_dir, f"annotations/{split}.txt"), 'r') as f:
            for i, line in enumerate(f.readlines()):
                qid, vid, duration, start, end, text = line.strip().split(" | ")  #text也就是query
                qid = int(qid)

                assert float(start) < float(end), \
                    "Wrong timestamps for {}: start >= end".format(qid)
                
                if vid not in self.v2dur:   #一个视频只有一个duration，表示整个视频的持续长度
                    self.v2dur[vid] = float(duration)
                self.q2v[qid] = {  #一个query对应一个视频以及该视频的持续时间和查询文本本身，
                    "vid": vid,
                    "duration": float(duration),
                    "timestamps": [float(start), float(end)],
                    "text": text.lower()
                }
                self.v2q[vid].append(qid)  #一个视频对应多个query，采用append放，只保存qid，qid字典有更详细的信息
        
        self.samples = list()
        #一共20个epoch,通过epoch的迭代，最终每个视频都会重复读取epoch次
        for i_epoch in range(epochs): #每迭代一轮，在Dataset初始化里面使用，在训练的时候没有显式使用
            batches = list()
            #如果epochs为1，那么就是只对每个视频进行一次迭代，读到的就是不重复的数据集
            for vid, qids in self.v2q.items(): #对每一个视频而言，视频和query是一对多的关系
                cqids = qids.copy()  #cqids是对每一个视频的
                if self.split == "train": #在训练模式下，才会进行填充
                    random.shuffle(cqids)  #每个视频的query都是随机打乱的
                    if len(cqids) % batch_size != 0: #将每个batch的查询样本都补足到batch_size:7上
                        pad_num = batch_size - len(cqids) % batch_size
                        #意思是每一个视频的query数目都要和batch_size对齐
                        cqids = cqids + cqids[:pad_num]  
                #表示这个视频的query总的数目占多少个batch
                #在测试的时候bsz100万，因此取上整，结果为1，测试时由于不需要进行pad，因此cqids的长度就是测试数据一个视频本身对应的数据query数目
                steps = np.math.ceil(len(cqids) / batch_size) #训练的时候就是整除，本身就对齐了
                for j in range(steps):
                    #每一条”样本“都有一个视频和batch_size个query,一共有多少个重复的视频取决于step数和vid数目
                    #batches的元素个数不确定，取决于每个视频对应了多少个query(不固定)，但每个元素都是一个视频和bsz个query组成
                    #包含的是所有的视频和所有的query,qids会有重复的元素，在训练模式下，由于mini-batch的设置，同一个视频会出现多次
                    #但是在测试模式下，直接一个视频对应其所有的query
                    batches.append({"vid": vid, "qids": cqids[j*batch_size:(j+1)*batch_size]})
            if self.split == "train": 
                random.shuffle(batches)  #每条样本中是视频和其对应的bsz个query,但是在每个batch内，视频样本之间是随机打乱的
            self.samples.extend(batches) #所有视频按照query数目为bsz大小综合而得
        # self.vfeat_path = os.path.join(self.data_dir, "features/CLIP_frames_features_5fps.h5")
        
        #使用了绝对地址来读取视频特征，数据特征在另一个比较大的挂载硬盘上
        self.vfeat_path = '/mnt/hdd1/zhulu/mad/CLIP_B32_frames_features_5fps.h5'
        self.qfeat_path = os.path.join(self.data_dir, "features/CLIP_language_sentence_features.h5") #文本使用的是句子级别特征
        if pre_load:  #初始化为false
            with h5py.File(self.vfeat_path, 'r') as f:
                self.vfeats = {m: np.asarray(f[m]) for m in self.v2q.keys()}
            with h5py.File(self.qfeat_path, 'r') as f:
                self.qfeats = {str(m): np.asarray(f[str(m)]) for m in self.q2v.keys()}
        else:
            #不要预训练的自己编码？ 不要一次性batch拿视频特征？？在getitem里面一个一个拿
            self.vfeats, self.qfeats = None, None 
        self.fps = 5.0  #视频本身特征的采样频率，1秒5帧


    def __len__(self):
        #Dataset所有的样本数，并且每个视频都重复了epoch次，并且还打乱了顺序
        #Samples和视频是一对一的关系，但是旗下包含多个query以及timestamp信息
        return len(self.samples)  


    def __getitem__(self, idx):
        #batch和epoch是在dataset init中使用的，samples已经是每个视频数据重复epoch次的集合了
        #samples中的每个数据，包括一个视频和bsz个query,dataloader中的batch_size是1表示拿到一个视频（以及其对应的bsz个query）
        vid = self.samples[idx]["vid"]  #从self.samples中拿数据
        #在dataset init的时候确定了，一个视频对应bsz个query
        #对于测试集来说，每个video对应的qid数目可能是不一样的
        qids = self.samples[idx]["qids"]  
        duration = self.v2dur[vid]  
        if not self.vfeats:  #init中pre_load为false，因此这里是None
            #读取第一个数据之后，就有了vfeats特征，之后的数据就不用再读取文件了，并且该文件的读取也比较快，不是时间耗费的主要原因
            self.vfeats = h5py.File(self.vfeat_path, 'r') 
        ori_video_feat = np.asarray(self.vfeats[vid])
        ori_video_length, feat_dim = ori_video_feat.shape 

        querys = {
            "texts": list(),
            "query_feats": list(),
            "query_masks": list(),
            "anchor_masks": list(),
            "starts": list(),
            "ends": list(),
            "overlaps": list(),
            "timestamps": list(),
        }
        
        similarity_threshold = 0.9
        #得到视频的事件表示索引编号，这部分得到的是numpy的结果
        v_cluster_event = region_growing_event_clustering(ori_video_feat,similarity_threshold)
        v_cluster_event=events_modify(v_cluster_event)
        #np.unique 可以直接实现事件的重新编号，得到非递减的编号，解决了事件编号不连续的问题
        unique_events, v_cluster_event = np.unique(v_cluster_event, return_inverse=True) #也可以直接用这个来作为新的事件表示了，统一移动了
        
        for qid in qids: #一个视频对应batch_size个query
            text = self.q2v[qid]["text"]  #q2v是一对一
            timestamps = self.q2v[qid]["timestamps"]  #都是一一对应的
            if not self.qfeats:
                self.qfeats = h5py.File(self.qfeat_path, 'r')
            query_feat = np.asarray(self.qfeats[str(qid)]) #在预处理文件中，一个query的特征，句子级别[cls]，没有单词个数维度
            querys["texts"].append(text) #一个视频对应多个query所以最后是列表的形式，每个元素是一个query
            querys["query_feats"].append(torch.from_numpy(query_feat))
            querys["timestamps"].append(torch.FloatTensor(timestamps))
        
        instance = {
            "vid": vid,
            "duration": float(duration), #duration就只有一个
            #因为video特征是经过pad了的，所以有的视频最后一部分是0，但是不影响
            "video_feats": torch.from_numpy(ori_video_feat).unsqueeze(0).float(),   #0号维度为 1 batch
            # 只用传递事件相对于视频帧的同维度序列，0开始非递减,并且连续
            "video_events": torch.from_numpy(v_cluster_event).unsqueeze(0).float(),  #保持和视频帧维度一致，只是没有特征维度
            "qids": qids,
            "texts":querys["texts"],
            #bsz,query_length,hidden_dim
            #对于测试集来讲，每个视频对应的query bsz不是固定的
            #但由于视频层级上的bsz是1，因此不需要涉及对齐操作，一次性就拿了所有的query，没有mini-batch的概念
            "query_feats": torch.stack(querys["query_feats"], dim=0).float(), 
            "timestamps": torch.stack(querys["timestamps"], dim=0)  #在第0个维度上进行叠加
        }
        return instance  


    def pad(self, arr, pad_len):
        new_arr = np.zeros((pad_len, ), dtype=float)
        new_arr[:len(arr)] = arr
        return new_arr 


    @staticmethod
    def collate_fn(data):
        all_items = data[0].keys()
        no_tensor_items = ["vid", "duration", "qids", "texts"]

        batch = {k: [d[k] for d in data] for k in all_items}
        for k in all_items:
            if k not in no_tensor_items:
                batch[k] = torch.cat(batch[k], dim=0)
        
        return batch



if __name__ == "__main__":
    import yaml
    with open("conf/soonet_mad.yaml", 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
        print(cfg)

    mad_dataset = MADDataset("train", cfg)    
    data_loader = data.DataLoader(mad_dataset, 
                            batch_size=1,
                            num_workers=4,
                            shuffle=False,
                            collate_fn=mad_dataset.collate_fn,
                            drop_last=False
                        )

    for i, batch in enumerate(data_loader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print("{}: {}".format(k, v.size()))
            else:
                print("{}: {}".format(k, v))
        break