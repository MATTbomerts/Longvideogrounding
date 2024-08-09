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
from ..utils import compute_overlap


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
        self.snippet_length = cfg.MODEL.SNIPPET_LENGTH  #10 对应论文中C0
        self.scale_num = cfg.MODEL.SCALE_NUM  #4  对应论文中尺度数
        self.max_anchor_length = self.snippet_length * 2**(self.scale_num - 1)   #对应一维卷积的长度？？请见下文
        if split == "train":  #yaml文件中的两个配置在数据集初始化时使用，而不是在dataloader和训练代码中使用
            epochs = cfg.TRAIN.NUM_EPOCH
            batch_size = cfg.TRAIN.BATCH_SIZE
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
                self.q2v[qid] = {  #一个query对应一个视频以及该视频的持续时间，
                    "vid": vid,
                    "duration": float(duration),
                    "timestamps": [float(start), float(end)],
                    "text": text.lower()
                }
                self.v2q[vid].append(qid)  #一个视频对应多个query，采用append放
        
        self.samples = list()
        #一共20个epoch,通过epoch的迭代，最终每个视频都会重复读取epoch次
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
                    #每一条”样本“都有一个视频和batch_size个query,一共有多少个重复的视频取决于step数和vid数目
                    #batches的元素个数不确定，取决于每个视频对应了多少个query(不固定)，但每个元素都是一个视频和bsz个query组成
                    #包含的是所有的视频和所有的query,qids会有重复的元素，在训练模式下，由于mini-batch的设置，同一个视频会出现多次
                    #但是在测试模式下，直接一个视频对应其所有的query
                    batches.append({"vid": vid, "qids": cqids[j*batch_size:(j+1)*batch_size]})
            if self.split == "train": 
                random.shuffle(batches)
            self.samples.extend(batches) #所有视频按照query数目为bsz大小综合而得
        # self.vfeat_path = os.path.join(self.data_dir, "features/CLIP_frames_features_5fps.h5")
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
        ori_video_length, feat_dim = ori_video_feat.shape 
        #计算填充后的视频长度，使其成为 max_anchor_length 的整数倍，结合一维卷积滑动窗口的范围来处理？
        pad_video_length = int(np.math.ceil(ori_video_length / self.max_anchor_length) * self.max_anchor_length)
        pad_video_feat = np.zeros((pad_video_length, feat_dim), dtype=float)
        pad_video_feat[:ori_video_length, :] = ori_video_feat
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
        scale_boundaries = [0]
        for qid in qids: #一个视频对应batch_size个query
            text = self.q2v[qid]["text"]  #q2v是一对一
            timestamps = self.q2v[qid]["timestamps"]  #都是一一对应的
            if not self.qfeats:
                self.qfeats = h5py.File(self.qfeat_path, 'r')
            query_feat = np.asarray(self.qfeats[str(qid)]) #在预处理文件中，一个query的特征，句子级别[cls]，没有单词个数维度
            query_length = query_feat.shape[0] #句子级别的特征为什么会有length维度？
            #虽然在读数据时，计算了query_mask，但是在后面的计算中并没有用到该mask，sentence-based mask是没有用的
            query_mask = np.ones((query_length, ), dtype=float)

            # generate multi-level groundtruth 每个尺度的掩码、开始、结束、每个尺度下Anchor与该query的标注区间的重叠程度
            masks, starts, ends, overlaps = list(), list(), list(), list()
            for i in range(self.scale_num):   #4，针对每一个scale进行处理，一个scale一次性处理
                anchor_length = self.snippet_length * 2**i #长度变化范围为10, 20, 40, 80  这里得到的anchor
                nfeats = math.ceil(ori_video_length / anchor_length) #表示原本视频应该有多少个anchor
                #因为是将长视频切换为不同的尺度，因此每一个尺度下，固定大小的anchor都有其起始和结束时刻来覆盖整个视频
                s_times = np.arange(0, nfeats).astype(np.float32) * (anchor_length / self.fps)
                #e_times 则是 s_times 往后移动一个anchor的时间，且不超过视频时长 duration
                e_times = np.minimum(duration, np.arange(1, nfeats + 1).astype(np.float32) * (anchor_length / self.fps))
                candidates = np.stack([s_times, e_times], axis=1) #candidates是当前尺度下每个anchor的起始和结束，不同尺度candidate数目不同
                
                #得到的是在该query下的timestamp和分多尺度之后的anchor的重叠程度，和视频duration无关
                overlap = compute_overlap(candidates.tolist(), timestamps)  
                #这个mask是干啥用的？表示有多少个实际有效的anchor吗？
                mask = np.ones((nfeats, ), dtype=int)
                #加上pad的视频帧之后，一共有多少个anchor，pad_nfeats在每个尺度下都会变化
                pad_nfeats = math.ceil(pad_video_length / anchor_length) 
                ends.append(self.pad(e_times, pad_nfeats)) #pad  np.zeros((pad_nfeats,))实际上操作后是一个(pad_nfeats,)的数组
                starts.append(self.pad(s_times, pad_nfeats))
                #pad之后都是(pad_nfeats,)的数组,因此后面不同尺度的concatenate操作是得到(多尺度之和，)，虽然每个尺度下的anchor数目不同，但是总的加起来是确定不变的
                overlaps.append(self.pad(overlap, pad_nfeats))  
                masks.append(self.pad(mask, pad_nfeats))

                if len(scale_boundaries) != self.scale_num + 1:  #一共5个数值，区间数为4
                    #用于记录每个尺度（scale）下的anchor数量的累积和(包括填充)，从而帮助在后续处理中方便地定位每个尺度的 anchor 范围
                    scale_boundaries.append(scale_boundaries[-1] + pad_nfeats) #这个值是累加的，第二层的anchor数量是第一层的anchor数量加上第二层的anchor数量
            
            #处理完一个query之后
            starts = np.concatenate(starts, axis=0)  #axis等于0，表示列数不变，行数拼接（只要保证列维度相同即可）；行数表示多少个尺度
            # print("starts shape: {}, vid: {}".format(starts.shape,self.q2v[qid]["vid"]))
            ends = np.concatenate(ends, axis=0) 
            overlaps = np.concatenate(overlaps, axis=0) #行数表示所有尺度下一共有多少个anchor，列数表示每个anchor的重叠程度
            masks = np.concatenate(masks, axis=0)

            #这一部分也没有scale_boundaries的处理（意思是所有query都使用同一个scale_boundaries）
            #如果是因为都是针对同一个视频的处理，那starts，ends应该也是一样的，不用每个query都有吧
            querys["texts"].append(text) #一个视频对应多个query所以最后是列表的形式，每个元素是一个query
            querys["query_feats"].append(torch.from_numpy(query_feat))
            querys["query_masks"].append(torch.from_numpy(query_mask))
            querys["anchor_masks"].append(torch.from_numpy(masks))
            querys["starts"].append(torch.from_numpy(starts)) #将该视频对应的bsz个query的所有尺度下的anchor的开始时间存入列表
            querys["ends"].append(torch.from_numpy(ends))
            querys["overlaps"].append(torch.from_numpy(overlaps))  #每个query下所有尺度下所有的anchor的重叠
            querys["timestamps"].append(torch.FloatTensor(timestamps))
        
        instance = {
            "vid": vid,
            "duration": float(duration), #duration就只有一个
            #因为video特征是经过pad了的，所以有的视频最后一部分是0，但是不影响
            "video_feats": torch.from_numpy(pad_video_feat).unsqueeze(0).float(),   #0号维度为 1 batch
            #为什么scale_boundaries没有根据query batch变化而变化？
            # "scale_boundaries": torch.LongTensor(scale_boundaries).repeat(32, 1), #repeat了32次，原本没有进行batch维度的扩充
            "scale_boundaries": torch.LongTensor(scale_boundaries),
            "qids": qids,
            "texts":querys["texts"],
            #bsz,query_length,hidden_dim
            #对于测试集来讲，每个视频对应的query bsz不是固定的
            # 但由于视频层级上的bsz是1，因此不需要涉及对齐操作，一次性就拿了所有的query，没有mini-batch的概念
            "query_feats": torch.stack(querys["query_feats"], dim=0).float(), 
            "query_masks": torch.stack(querys["query_masks"], dim=0).float(),
            "anchor_masks": torch.stack(querys["anchor_masks"], dim=0),
            "starts":  torch.stack(querys["starts"], dim=0), #bsz, num_all_anchors
            "ends": torch.stack(querys["ends"], dim=0),
            "overlaps": torch.stack(querys["overlaps"], dim=0), #32个query的所有尺度下所有的anchor的重叠
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