import numpy as np
import torch
import torch.nn as nn 

from .blocks import *
from .swin_transformer import SwinTransformerV2_1D
from .loss import *
from ..utils import fetch_feats_by_index, compute_tiou, AttentionModule
import time


class SOONet(nn.Module):

    def __init__(self, cfg):
        
        super().__init__()
        nscales = cfg.MODEL.SCALE_NUM #4
        hidden_dim = cfg.MODEL.HIDDEN_DIM
        snippet_length = cfg.MODEL.SNIPPET_LENGTH  #10
        enable_stage2 = cfg.MODEL.ENABLE_STAGE2  #参数为true
        stage2_pool = cfg.MODEL.STAGE2_POOL  #在论文中对应计算内部帧和文本相似度之后的average操作
        stage2_topk = cfg.MODEL.STAGE2_TOPK  #100
        topk = cfg.TEST.TOPK  #100

        #先开始写死抽象事件表示的参数
        self.abstract_encoder = AttentionModule(15, 512)
        self.video_encoder = SwinTransformerV2_1D(
                                patch_size=snippet_length, 
                                in_chans=hidden_dim, 
                                embed_dim=hidden_dim, 
                                depths=[2]*nscales, 
                                num_heads=[8]*nscales,
                                window_size=[64]*nscales,   #参数是[64,64,64,64]
                                mlp_ratio=2., 
                                qkv_bias=True,
                                drop_rate=0., 
                                attn_drop_rate=0., 
                                drop_path_rate=0.1,
                                norm_layer=nn.LayerNorm, 
                                patch_norm=True,
                                use_checkpoint=False, 
                                pretrained_window_sizes=[0]*nscales
                            )
        
        self.q2v_stage1 = Q2VRankerStage1(nscales, hidden_dim)  #anchor rank optimization 4，512
        self.v2q_stage1 = V2QRankerStage1(nscales, hidden_dim)  #query rank optimization
        if enable_stage2:
            self.q2v_stage2 = Q2VRankerStage2(nscales, hidden_dim, snippet_length, stage2_pool)
            self.v2q_stage2 = V2QRankerStage2(nscales, hidden_dim)
        self.regressor = BboxRegressor(hidden_dim, enable_stage2)
        self.rank_loss = ApproxNDCGLoss(cfg)
        self.reg_loss = IOULoss(cfg)

        self.nscales = nscales
        self.enable_stage2 = enable_stage2
        self.stage2_topk = stage2_topk  #在测试阶段才使用了top-k 100
        self.cfg = cfg
        self.topk = topk  # 100
        self.enable_nms = cfg.MODEL.ENABLE_NMS #false


    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def forward_train(self, 
                      query_feats=None,
                      query_masks=None,
                      video_feats=None,  #在dataloaer中给出了0号维度bsz值为1 
                      video_events=None,
                      timestamps=None,
                      **kwargs):
        
        batch_size = video_feats.size(0)
        hidden_dim = video_feats.size(2)

        # 初始化一个列表用于存储每个事件的平均特征，此处的操作还不涉及到批量化
        event_avg_feats_list = []

        for b in range(batch_size):  #一开始简单操作，就一个视频起手
            # 获取当前批次的视频特征和事件
            current_video_feats = video_feats[b]  # 形状为 (frames, hidden_dim)
            current_video_events = video_events[b]  # 形状为 (frames,)
            
            # 获取唯一的事件ID以及它们的反向索引，拿到的index是从0，1，2...连续开始的，不对应于事件本身的编号
            unique_events, inverse_indices = torch.unique(current_video_events, return_inverse=True) #也可以直接用这个来作为新的事件表示了，统一移动了
            #[0,2,4,5] [0,0,0,1,1,1,1,2,3,3]
            # 在这个位置就可以得到每个事件的时间区间，按照fps 5的参数条件下，需要得到的是事件的start和end，然后得到边界的偏移(或者其他)
            
            # 初始化一个矩阵用于存储每个事件的平均特征
            event_avg_feats = torch.zeros((unique_events.size(0), hidden_dim))
            
            # 计算每个事件的平均特征
            event_timestamps=list()
            for i in range(unique_events.size(0)):
                frame_rate=5
                #得到每个事件区间的Anchor特征
                event_avg_feats[i] = current_video_feats[inverse_indices == i].mean(dim=0)
                #得到每个事件的开始和结束时间
                event_start_index = (inverse_indices == i).nonzero(as_tuple=True)[0][0]
                event_end_index = (inverse_indices == i).nonzero(as_tuple=True)[0][-1]
                event_start_time=event_start_index/frame_rate
                event_end_time=event_end_index/frame_rate
                event_timestamps.append(torch.tensor([event_start_time,event_end_time]))
                
            event_timestamps=torch.stack(event_timestamps,dim=0)  #得到tensor表示的时间戳，此处都是所有的事件
            # 将结果添加到列表中
            event_avg_feats_list.append(event_avg_feats)


        # 这里最终的输出形状将为 (batch_size, num_events, hidden_dim)，在操作中bsz值为1 ,final_event的维度和unique_events的维度相关
        final_event_anchor_feats = torch.stack(event_avg_feats_list)
        final_event_anchor_feats=final_event_anchor_feats.to(query_feats.device)

        final_event_anchor_feats = final_event_anchor_feats.squeeze(0) 
        
        # 对 final_event_anchor_feats 和 query_feats 进行归一化
        final_event_anchor_feats_norm = F.normalize(final_event_anchor_feats, p=2, dim=-1)  # (num_anchors, hidden_dim)
        query_feats_norm = F.normalize(query_feats, p=2, dim=-1)  # (batch_query, hidden_dim)
        #使用余弦相似度计算其值能达到在【-1，1】之间，更好去判断相似的程度 ，维度为【batch_query,event anchor】
        #此处得到的是所有事件和查询之间的相似度，进一步才根据相似度选择出需要学习的事件
        similarity_matrix = torch.nn.functional.cosine_similarity(query_feats_norm.unsqueeze(1), final_event_anchor_feats_norm.unsqueeze(0), dim=-1)
        
        # todo: baseline的时候先不进行事件之间的聚集，再后面的版本中再进行聚集
        #根据相似度阈值选择出对于每个查询需要在视频中使用学习的事件
        event_cluster_threshold = 0.03
        selected_columns = similarity_matrix > event_cluster_threshold #得到的依然是索引，不是真实事件ID，表示的是第几个事件

        # 结果是一个列表，每个元素是一个张量，包含每一行中满足条件的列索引，此处还是需要每个事件对应的时间区间,是有重复值的，不同的样本查询
        selected_indices = [torch.nonzero(row).squeeze(1) for row in selected_columns]

        hit_indices = list()  #表示所有的查询选择出来的哪些事件
        for indice in selected_indices:
            hit_indices.extend(indice.tolist())
            
        hit_indices = list(set(hit_indices))  #只是相似度达到xx,直接全选出来，得到的就是选择的事件，先不进行后续的事件聚类操作
        
        #将事件转换为抽象固定长度，先简便版本，直接for循环针对每个事件操作，后续考虑基于长度的pad,mask方法
        #目前还差时间信息
        abstract_events = list() #就是从hit_indices中选择出来的事件
        for indice in hit_indices:  #hit_indices是所有的事件的索引，因此操作无误
            event_feat = video_feats[0][inverse_indices==indice]
            abstract_features = self.abstract_encoder(event_feat)  #抽象事件的表示
            abstract_events.append(abstract_features)
        
        """不再是所有的事件，索引不再对应，有一个压缩映射效果"""
        abstract_events=torch.stack(abstract_events,dim=0)  #维度为(event select num, abstract_query,hidden_dim)
        select_event_timestamps=event_timestamps[torch.tensor(hit_indices)]  #得到选择的事件的时间戳，用于后面将边界偏移转换为绝对时间
        #上述操作之后，hit_indices中非连续的事件索引就变成了select_event_timestamps中连续的索引
        select_duration=select_event_timestamps[:,1]-select_event_timestamps[:,0]  #得到选择的事件的持续时间
        
        #选择的事件预测结果，不是所有事件。bbox_bias会得到(bsz_query,total_num_anchors,2)的结果，已经能有正常结果了
        #下面四个是一一对应的
        bbox_bias = self.regressor(abstract_events, query_feats,mode="train") #结合context-based和content-based的特征进行bbox回归
        select_event_timestamps=select_event_timestamps.to(bbox_bias.device)
        select_duration=select_duration.to(bbox_bias.device)
        pred_timestamps = select_event_timestamps.unsqueeze(0) + bbox_bias * select_duration.unsqueeze(0).unsqueeze(-1)
        #要去计算原本聚类得到的事件到底哪些和真实查询的标注存在相交，用这部分的事件得到的预测值进行损失的计算才对(不然选择无关的片段强行将时间对齐，学习到的内容将有问题)
        #最终可能就少数几个事件和真实标注的区间相关，没有多尺度
        event_overlaps_pred=list() #是事件的时间戳，不是标注的时间戳，每个查询损失对应的事件个数不一定一样，但是可以统一转换为mask的方式去做
        #要将timestamps由原本的s为单位转换为帧才能确定对应的事件，round得到的是取整后的浮点数，还不是integer形式
        frame_timestamps=torch.round(timestamps*5).int() #真实标注的帧区间，有的秒数可能不足1s，但是帧数肯定大于一帧，取round应该没啥问题

        #对每个查询而言
        for query_index,frame_timestamp in enumerate(frame_timestamps):
            #针对每个查询对应的时间戳，tensor(594, device='cuda:0')两个不会因为取set集合而去重，只能torch.unique
            overlap_events=torch.unique(inverse_indices[frame_timestamp[0]:frame_timestamp[1]+1])  #得到真实标注区间对应的事件索引
            #所有的事件和查询的标注进行相交比较，大于选择的事件数目
            overlap_event_predict=[]
            for overlap_event in overlap_events:  #所有的事件，hit_indices数目不是所有的，但是编号是和所有的事件一一对应
                if overlap_event in hit_indices:  #overlap_event是一个一个进行遍历计算的
                    #选择得到的事件、预测结果的索引和hit_indices存在压缩对应的关系，位序编号保持一致
                    #原本在hit_indices中第1个元素为6(表示第6个事件)，在pred_timestamps中将会变为第一个元素值，对应的是所有事件的第6个
                    #因此应该找hit_indices中值(事件)的索引，在选择后的事件中使用该索引得到该事件的预测值
                    overlap_index = (torch.tensor(hit_indices).to(overlap_event.device) == overlap_event).nonzero(as_tuple=True)[0]
                    overlap_event_predict.append(pred_timestamps[query_index][overlap_index].squeeze())#得到的是绝对的时间，如果不使用squeeze会多一个维度
                else: #会存在真实标注的事件没有被选择的情况，这种情况另外计算赋值其他,强制给一个损失还算比较大
                    overlap_event_predict.append(torch.tensor([0,0],device=bbox_bias.device)) 
            event_overlaps_pred.append(torch.stack(overlap_event_predict)) 

        loss_dict=self.loss(event_overlaps_pred,timestamps)
        print("ok")
        return loss_dict

    def forward_test(self,
                     query_feats=None,
                     video_feats=None,
                     video_events=None,
                     timestamps=None,
                     **kwargs):
        """

        Args:
            query_feats (_torch_, optional): _description_. (all_querys,hidden_dim)一个视频的所有query特征
            query_masks (_torch_, optional): _description_. 句子级别没有用
            video_feats (_type_, optional): _description_. (1,frames,hidden_dim)视频特征，
            start_ts (_type_, optional): _description_. Defaults to None.
            scale_boundaries (_type_, optional): _description_. Defaults to None.

        Returns:                                  
            merge_scores  _torch_: _description_: 0维度表示该视频对应的query总数，1维度表示topk个anchor对应的单值分数
            merge_bboxes  _torch_: _description_: 0维度表示该视频对应的query总数，1维度表示topk个anchor个数，2维度表示每个anchor对应的开始结束时间
        """
        #在测试的时候就没法使用overlap
        test_gpu_st=time.time()
        test_gpu=list()
        test_cpu=list()
        
        video_events=video_events.squeeze() #先考虑bsz为1，不然unique得到的inverse_indices会有问题
        unique_events, inverse_indices = torch.unique(video_events, return_inverse=True) #也可以直接用这个来作为新的事件表示了，统一移动了
        #[0,2,4,5] [0,0,0,1,1,1,1,2,3,3]
        # 在这个位置就可以得到每个事件的时间区间，按照fps 5的参数条件下，需要得到的是事件的start和end，然后得到边界的偏移(或者其他)
        
        # 初始化一个矩阵用于存储每个事件的平均特征
        hidden_dim=video_feats.size(-1)
        event_avg_feats = torch.zeros((unique_events.size(0), hidden_dim))
        
        # 计算每个事件的平均特征
        event_timestamps,event_avg_feats_list=list(),list()
        for i in range(unique_events.size(0)):
            frame_rate=5
            #得到每个事件区间的Anchor特征
            event_avg_feats[i] = video_feats[0][inverse_indices == i].mean(dim=0)
            #得到每个事件的开始和结束时间
            event_start_index = (inverse_indices == i).nonzero(as_tuple=True)[0][0]
            event_end_index = (inverse_indices == i).nonzero(as_tuple=True)[0][-1]
            event_start_time=event_start_index/frame_rate
            event_end_time=event_end_index/frame_rate
            event_timestamps.append(torch.tensor([event_start_time,event_end_time]))
            
        event_timestamps=torch.stack(event_timestamps,dim=0)  #得到tensor表示的时间戳，此处都是所有的事件
        # 将结果添加到列表中
        event_avg_feats_list.append(event_avg_feats)
        # 这里最终的输出形状将为 (batch_size, num_events, hidden_dim)，在操作中bsz值为1 ,final_event的维度和unique_events的维度相关
        final_event_anchor_feats = torch.stack(event_avg_feats_list)
        final_event_anchor_feats=final_event_anchor_feats.to(query_feats.device)
        final_event_anchor_feats = final_event_anchor_feats.squeeze(0) 
        final_event_anchor_feats_norm = F.normalize(final_event_anchor_feats, p=2, dim=-1)  # (num_anchors, hidden_dim)
        query_feats_norm = F.normalize(query_feats, p=2, dim=-1)  # (batch_query, hidden_dim)
        similarity_matrix = torch.nn.functional.cosine_similarity(query_feats_norm.unsqueeze(1), final_event_anchor_feats_norm.unsqueeze(0), dim=-1)
        
        #在测试的时候原设想的逻辑是通过模块先来判断事件是否有query检测，然后再从其中进行预测
        #在baseline中的实现是通过相似度来进行选择，都没进行事件的融合，那就直接从相似度分数中选择前top-k个吧，但是需要对相似度分数进行学习，确保这个相似度分数是有效的区分标准
        #对每个查询选择相似度最高的top-k个事件
        #在测试的环境下使用的是一个视频和其对应的所有的查询，在dataloader中如此设置了
        _, top_indices = torch.topk(similarity_matrix, self.stage2_topk, dim=1)
        #直接在整个矩阵中得到不重复的元素，得到这一批次的查询共享的事件,每个查询选择100个，总的事件可能就比较多
        hit_indices = torch.unique(top_indices)
        
        abstract_events = list() #就是从hit_indices中选择出来的事件
        for indice in hit_indices:  #hit_indices是所有的事件的索引，因此操作无误
            event_feat = video_feats[0][inverse_indices==indice]
            abstract_features = self.abstract_encoder(event_feat)  #抽象事件的表示
            abstract_events.append(abstract_features)
        abstract_events=torch.stack(abstract_events,dim=0)  #维度为(event select num, abstract_query,hidden_dim)
        event_timestamps=event_timestamps.to(hit_indices.device)
        select_event_timestamps=event_timestamps[hit_indices]  #得到选择的事件的时间戳，用于后面将边界偏移转换为绝对时间
        #上述操作之后，hit_indices中非连续的事件索引就变成了select_event_timestamps中连续的索引
        select_duration=select_event_timestamps[:,1]-select_event_timestamps[:,0]  #得到选择的事件的持续时间
        bbox_bias = self.regressor(abstract_events, query_feats,mode="test") #结合context-based和content-based的特征进行bbox回归
        select_event_timestamps=select_event_timestamps.to(bbox_bias.device)
        select_duration=select_duration.to(bbox_bias.device)
        #pred_timestamps就是最终选择出来的top-k个事件预测得到的结果
        pred_timestamps = select_event_timestamps.unsqueeze(0) + bbox_bias * select_duration.unsqueeze(0).unsqueeze(-1)
        return pred_timestamps
        # return merge_scores, merge_bboxes

    def loss(self, 
             bbox_bias, #经过与真实片段存在相交的事件的预测值，列表数据一共timestamps.size(0)个元素
             timestamps,
             ):
        
        iou_loss =0.0

        # sbias = bbox_bias[:, :, 0]
        # ebias = bbox_bias[:, :, 1]
        # duration = ends - starts
        # #此处的duration不是整个视频的duration，而是这个anchor区间的duration
        # pred_start = starts + sbias * duration  
        # pred_end = ends + ebias * duration
        

        #要从pred_start和pred_end中选择出与真实标注有重叠的anchor，然后计算损失

        #只对那些与真实标注有重叠的anchor进行计算损失(要选择实际要计算损失的对象)
        # if self.cfg.MODEL.ENABLE_STAGE2:  
        #     iou_mask = stage2_overlaps > self.cfg.LOSS.REGRESS.IOU_THRESH
        # else:
        #     iou_mask = overlaps > self.cfg.LOSS.REGRESS.IOU_THRESH
        
        iou_loss = self.reg_loss(bbox_bias, timestamps)

        total_loss = self.cfg.LOSS.REGRESS.WEIGHT * iou_loss

        loss_dict = {
            "reg_loss": iou_loss,
            "total_loss": total_loss
        }
        return loss_dict


    def filter_anchor_by_iou(self, gt, pred):
        indicator = (torch.sum((gt > self.cfg.LOSS.V2Q.MIN_IOU).float(), dim=0, keepdim=False) > 0).long()
        moment_num = torch.sum(indicator)
        _, index = torch.sort(indicator, descending=True)
        index = index[:moment_num]
        gt = torch.index_select(gt, 1, index).transpose(0, 1)
        pred = torch.index_select(pred, 1, index).transpose(0, 1)
        return gt, pred


    def nms(self, pred, thresh=0.3, topk=5):
        nms_res = list()
        mask = [False] * len(pred)
        for i in range(len(pred)):
            f = pred[i].copy()
            if not mask[i]:
                nms_res.append(f)
                if len(nms_res) >= topk:
                    break
                for j in range(i, len(pred)):
                    tiou = compute_tiou(pred[i], pred[j])
                    if tiou > thresh:
                        mask[j] = True
        del mask
        return nms_res