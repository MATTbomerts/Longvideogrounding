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
from src.utils import compute_overlap
import time


# def calculate_similarity(feature1, feature2):
#     dot_product = np.dot(feature1, feature2)
#     norm_a = np.linalg.norm(feature1)  #L2范数，表示向量的长度
#     norm_b = np.linalg.norm(feature2)
#     similarity = dot_product / (norm_a * norm_b) #向量积除以向量长度的乘积等于相似度
#     return similarity

# # 根据相似度进行区域生长
# def region_growing_event_clustering(features, similarity_threshold):
#     num_frames = len(features)
#     # 初始化事件标签为-1,一开始所有的帧都标记为没有归类所属事件，大小和帧数一样
#     events = np.zeros(num_frames, dtype=int) - 1  
#     current_event = 0  #事件的记号，依次递增表示第几个事件
    
#     for i in range(num_frames):
#         #如果当前帧没有标记为任何事件，防止当前帧在聚集事件之后将相邻帧也进行了标记，后续还要对相邻帧进行计算的重复开销
#         if events[i] == -1:  #第一帧处理是没有事件标记的
#             events[i] = current_event
#             queue = [i]
            
#             while queue:
#                 current_frame = queue.pop(0)
#                 #neighbor也只是帧索引序号，当前帧的左右相邻两帧，左右两帧进行遍历计算
#                 for neighbor in [current_frame - 1, current_frame + 1]:
#                     #相邻帧在合理的范围内，且没有被标记为任何事件
#                     if 0 <= neighbor < num_frames and events[neighbor] == -1:
#                         similarity = calculate_similarity(features[current_frame], features[neighbor])
#                         if similarity > similarity_threshold:
#                             events[neighbor] = current_event
#                             queue.append(neighbor)
#             current_event += 1
#     return events

class MADDataset(data.Dataset):
    
    def __init__(self, split, cfg, pre_load=False):
        super().__init__()
        self.split = split
        self.data_dir = cfg.DATA.DATA_DIR
        self.snippet_length = cfg.MODEL.SNIPPET_LENGTH  #10 对应论文中C0
        self.scale_num = cfg.MODEL.SCALE_NUM  #4  对应论文中尺度数
        self.max_anchor_length = self.snippet_length * 2**(self.scale_num - 1)
        if split == "train":  #yaml文件中的两个配置在数据集初始化时使用，而不是在dataloader和训练代码中使用
            epochs = cfg.TRAIN.NUM_EPOCH
            batch_size = cfg.TRAIN.BATCH_SIZE
        else:
            epochs = 1
            batch_size = 1000000
        
        self.q2v = dict()  #一对一、qid对应的样本，但不包含视频特征，只有视频id
        self.v2q = defaultdict(list) #一对多
        self.v2dur = dict()
        self.v2events = dict()
        with open(os.path.join(self.data_dir, f"annotations/{split}.txt"), 'r') as f:
            for i, line in enumerate(f.readlines()):
                qid, vid, duration, start, end, text = line.strip().split(" | ")  #text也就是query
                qid = int(qid)

                assert float(start) < float(end), \
                    "Wrong timestamps for {}: start >= end".format(qid)
                
                if vid not in self.v2dur:   #一个视频只有一个duration，表示整个视频的持续长度
                    self.v2dur[vid] = float(duration)
                self.q2v[qid] = {  #一个query对应一个视频以及该视频的持续时间，
                    "vid": vid,
                    "duration": float(duration),
                    "timestamps": [float(start), float(end)],
                    "text": text.lower()
                }
                self.v2q[vid].append(qid)  #一个视频对应多个query，采用append放
        
        # vfeat_path = '/mnt/hdd1/zhulu/mad/CLIP_B32_frames_features_5fps.h5'
        # vfeats = h5py.File(vfeat_path, 'r') 
        
        count=0
        # for vid, _ in self.v2q.items():
        #     ori_video_feat = np.asarray(vfeats[vid])
            # estime=time.time()
            # events = region_growing_event_clustering(ori_video_feat,similarity_threshold=0.75)
            # eetime=time.time()
            # print("event time:",eetime-estime)
            # self.v2events[vid] = events  #events只是事件的序号
            # count+=1
            # if(count%50==0):
            #     print("count:",count)
        print("dataset len v2e:",len(self.v2events))
        self.samples = list()
        #一共20个epoch,通过epoch的迭代，最终每个视频都会重复读取epoch次，在此处self.samples中还没有真正的拿到数据，是在getitem中获得的数据
        for i_epoch in range(epochs): #每迭代一轮，在Dataset初始化里面使用，在训练的时候没有显式使用
            batches = list()
            for vid, qids in self.v2q.items(): #对每一个视频而言，视频和query是一对多的关系
                cqids = qids.copy()  #cqids是对每一个视频的
                if self.split == "train": #在训练模式下，才会进行填充
                    random.shuffle(cqids)
                    if len(cqids) % batch_size != 0:
                        pad_num = batch_size - len(cqids) % batch_size
                        #意思是每一个视频的query数目都要和batch_size对齐
                        cqids = cqids + cqids[:pad_num]  
                #表示这个视频的query数目占多少个batch
                #在测试的时候bsz100万，因此取上整，结果为1，测试时由于不需要进行pad，因此cqids的长度就是测试数据一个视频本身对应的数据query数目
                steps = np.math.ceil(len(cqids) / batch_size)
                for j in range(steps):
                    #在sample中得到的样本只有vid，在getitem中才拿数据，将vid和特征对应到一起
                    batches.append({"vid": vid, "qids": cqids[j*batch_size:(j+1)*batch_size]})
            if self.split == "train": 
                random.shuffle(batches)
            self.samples.extend(batches) #所有视频按照query数目为bsz大小综合而得
        # self.vfeat_path = os.path.join(self.data_dir, "features/CLIP_frames_features_5fps.h5")
        self.vfeat_path = '/mnt/hdd1/zhulu/mad/CLIP_B32_frames_features_5fps.h5'
        self.qfeat_path = os.path.join(self.data_dir, "features/CLIP_language_sentence_features.h5") #文本使用的是句子级别特征
        
        print("dataset_init###################")
        if pre_load:  #初始化为false
            with h5py.File(self.vfeat_path, 'r') as f:
                self.vfeats = {m: np.asarray(f[m]) for m in self.v2q.keys()}
            with h5py.File(self.qfeat_path, 'r') as f:
                self.qfeats = {str(m): np.asarray(f[str(m)]) for m in self.q2v.keys()}
        else:
            self.vfeats, self.qfeats = None, None 
        self.fps = 5.0
        


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
        # events = np.asarray(self.v2events[vid])
        ori_video_length, feat_dim = ori_video_feat.shape 
        #计算填充后的视频长度，使其成为 max_anchor_length 的整数倍，结合一维卷积滑动窗口的范围来处理？
        pad_video_length = int(np.math.ceil(ori_video_length / self.max_anchor_length) * self.max_anchor_length)
        pad_video_feat = np.zeros((pad_video_length, feat_dim), dtype=float)
        pad_video_feat[:ori_video_length, :] = ori_video_feat
        
        querys = {
            "texts": list(),
            "query_feats": list(),
            "timestamps": list(),
        }
        for qid in qids: #一个视频对应batch_size个query
            text = self.q2v[qid]["text"]  #q2v是一对一
            timestamps = self.q2v[qid]["timestamps"]  #都是一一对应的
            if not self.qfeats:
                self.qfeats = h5py.File(self.qfeat_path, 'r')
            query_feat = np.asarray(self.qfeats[str(qid)]) #在预处理文件中，一个query的特征，句子级别[cls]，没有单词个数维度
            #这一部分也没有scale_boundaries的处理（意思是所有query都使用同一个scale_boundaries）
            #如果是因为都是针对同一个视频的处理，那starts，ends应该也是一样的，不用每个query都有吧
            querys["texts"].append(text) #一个视频对应多个query所以最后是列表的形式，每个元素是一个query
            querys["query_feats"].append(torch.from_numpy(query_feat))
            querys["timestamps"].append(torch.FloatTensor(timestamps))
        instance = {
            "vid": vid,
            "duration": float(duration), #duration就只有一个
            #因为video特征是经过pad了的，所以有的视频最后一部分是0，但是不影响
            
            "video_feats": torch.from_numpy(pad_video_feat).float(),   #0号维度为 1 batch
            "qids": qids,
            "texts":querys["texts"],
            # "events":torch.from_numpy(events).float(),
            #bsz,query_length,hidden_dim
            #对于测试集来讲，每个视频对应的query bsz不是固定的
            # 但由于视频层级上的bsz是1，因此不需要涉及对齐操作，一次性就拿了所有的query，没有mini-batch的概念
            "query_feats": torch.stack(querys["query_feats"], dim=0).float(), 
            "timestamps": torch.stack(querys["timestamps"], dim=0)  #在第0个维度上进行叠加
        }

        return instance  #一条样本，一个视频，batch_size个query以及对应的多尺度下的区间


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
        
        #在这里视频也是tensor特征
        
        return batch



if __name__ == "__main__":
    import yaml
    with open("conf/soonet_mad.yaml", 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
        print(cfg)

    mad_dataset = MADataset("train", cfg)    
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