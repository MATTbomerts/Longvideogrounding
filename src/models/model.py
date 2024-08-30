import numpy as np
import torch
import torch.nn as nn 

from .blocks import *
from .swin_transformer import SwinTransformerV2_1D
from .loss import *
from ..utils import fetch_feats_by_index, compute_tiou, AttentionModule,EventQueryClassifier,AttentionModule1
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
        self.abstract_encoder1 = AttentionModule1(15, 512)
        
        self.classifier = EventQueryClassifier(input_dim=2*512)  #2*hidden_dim是因为拼接了query和event的特征
        
        #region
        # self.video_encoder = SwinTransformerV2_1D(
        #                         patch_size=snippet_length, 
        #                         in_chans=hidden_dim, 
        #                         embed_dim=hidden_dim, 
        #                         depths=[2]*nscales, 
        #                         num_heads=[8]*nscales,
        #                         window_size=[64]*nscales,   #参数是[64,64,64,64]
        #                         mlp_ratio=2., 
        #                         qkv_bias=True,
        #                         drop_rate=0., 
        #                         attn_drop_rate=0., 
        #                         drop_path_rate=0.1,
        #                         norm_layer=nn.LayerNorm, 
        #                         patch_norm=True,
        #                         use_checkpoint=False, 
        #                         pretrained_window_sizes=[0]*nscales
        #                     )
        
        # self.q2v_stage1 = Q2VRankerStage1(nscales, hidden_dim)  #anchor rank optimization 4，512
        # self.v2q_stage1 = V2QRankerStage1(nscales, hidden_dim)  #query rank optimization
        # if enable_stage2:
        #     self.q2v_stage2 = Q2VRankerStage2(nscales, hidden_dim, snippet_length, stage2_pool)
        #     self.v2q_stage2 = V2QRankerStage2(nscales, hidden_dim)
        #endregion
        
        self.regressor = BboxRegressor(hidden_dim, enable_stage2)
        # self.rank_loss = ApproxNDCGLoss(cfg)
        self.reg_loss = IOULoss(cfg)
        self.classify_loss=nn.BCELoss()

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
                      video_feats=None,  #在dataloaer中给出了0号维度bsz值为1 
                      video_events=None,
                      timestamps=None,
                      **kwargs):
        
        train_st=time.time()
        batch_size = video_feats.size(0)  #bsz,frames,hidden_dim
        hidden_dim = video_feats.size(2)
        anchor_event_st=time.time()
        for b in range(batch_size):  #一开始简单操作，就一个视频起手
            # 获取当前批次的视频特征和事件
            current_video_feats = video_feats[b]  # 形状为 (frames, hidden_dim)
            current_video_events = video_events[b]  # 形状为 (frames,)
            #传入的current_video_events是连续的编号[0,0,0,1,1,1,1,2,3,3...]，输出得到的是连续的事件重新编号
            unique_events, inverse_indices = torch.unique(current_video_events, return_inverse=True) #也可以直接用这个来作为新的事件表示了，统一移动了
            
            """ 就算事件的平均特征anchor 以及事件的开始和结束事件 """
            event_avg_feats = torch.zeros((unique_events.size(0), hidden_dim))
            event_timestamps=list()
            #region
            # o_st=time.time()
            # for i in range(unique_events.size(0)):
            #     frame_rate=5
            #     #得到每个事件区间的Anchor特征
            #     event_avg_feats[i] = current_video_feats[inverse_indices == i].mean(dim=0)
            #     #得到每个事件的开始和结束时间
            #     event_start_index = (inverse_indices == i).nonzero(as_tuple=True)[0][0]
            #     event_end_index = (inverse_indices == i).nonzero(as_tuple=True)[0][-1]
            #     event_start_time=event_start_index/frame_rate
            #     event_end_time=event_end_index/frame_rate
            #     event_timestamps.append(torch.tensor([event_start_time,event_end_time]))
                
            # event_timestamps=torch.stack(event_timestamps,dim=0)  #得到tensor表示的时间戳，此处都是所有的事件
            # # 将结果添加到列表中
            # event_avg_feats_list.append(event_avg_feats)
            # event_avg_feats_list=torch.stack(event_avg_feats_list,dim=0).to(video_feats.device)
            # o_et=time.time()
            #endregion

            frame_rate=5
            # 1. 计算每个事件的平均特征
            event_avg_feats = torch.zeros((unique_events.size(0), current_video_feats.size(1)), device=current_video_feats.device)
            event_counts = torch.zeros(unique_events.size(0), device=current_video_feats.device)
            event_avg_feats = event_avg_feats.scatter_add(0, inverse_indices.unsqueeze(1).expand_as(current_video_feats), current_video_feats)
            event_counts = event_counts.scatter_add(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float32))
            event_avg_feats = event_avg_feats / event_counts.unsqueeze(1).clamp(min=1)
            # 计算平均值，防止除以零的情况，和上面的for循环的方法算出来有一定的浮点数精度问题，但是肉眼看不出来差别

            """计算每个事件的开始和结束时间，是视频所有帧"""
            frame_indices = torch.arange(inverse_indices.size(0), device=inverse_indices.device)

            # 使用scatter_reduce找到每个事件的最小和最大索引
            # 事件开始用最大的帧索引初始化，结束用最小的帧索引初始化
            event_start_indices = torch.full((unique_events.size(0),), fill_value=inverse_indices.size(0), device=inverse_indices.device)
            event_end_indices = torch.full((unique_events.size(0),), fill_value=0, device=inverse_indices.device)

            event_start_indices = event_start_indices.scatter_reduce(0, inverse_indices, frame_indices, reduce='amin')
            event_end_indices = event_end_indices.scatter_reduce(0, inverse_indices, frame_indices, reduce='amax')

            # 转换为开始和结束时间，这一步可以到后面再做
            # event_start_times = event_start_indices.float() / frame_rate
            # event_end_times = event_end_indices.float() / frame_rate
            
            event_start_times = event_start_indices  #得到的是帧索引，还没有变为时间
            event_end_times = event_end_indices

            # 将开始和结束时间拼接为 (events, 2) 的矩阵
            event_timestamps = torch.stack([event_start_times, event_end_times], dim=1)
            event_duration=event_timestamps[:,1]-event_timestamps[:,0]+1  # 加1才表示每个事件有多少帧,不是结束时间和开始时间做差,因此有特殊处理
            #知道每个事件的开始和结束时间(帧序)
        
        event_ob_st=time.time()
        #region
        # event_feats=[]
        #顺序得到每个基本事件所花的事件都达到0.11s
        # for i in range(unique_events.size(0)):
        #     event_st=event_timestamps[i][0] 
        #     event_end=event_timestamps[i][1]
            
        #     event_feat=video_feats[0][event_st:event_end+1]
        #     event_feats.append(event_feat)
        # event_ob_et=time.time()
        # print(f"event_ob_time: {event_ob_et-event_ob_st}")
        
        #根据长度来排序,从小到大的顺序
        # ab_feat=torch.zeros(unique_events.size(0),15,512,device=query_feats.device)
        # sorted_event_feats = sorted(event_feats, key=lambda x: x[1])
        # sorted_event_length=event_feats.sort(key=lambda x: x[1])  # 按帧数排序，从小到大进行事件按帧数进行排序，这个时间开销也不大啊，0.0002 
        #endregion
        #拿出前1000个事件,按照frames拼接起来,
        
        ab_st=time.time()
        
        mem_alloc_before = torch.cuda.memory_allocated()
        mem_reserved_before = torch.cuda.memory_reserved()
        
        abstract_feat=torch.zeros(unique_events.size(0),15,512,device=query_feats.device)
        mid_frame=video_feats[0].size(0)//2
        mid_event=inverse_indices[mid_frame]  #事件是从0开始的,因此一共包含1+mid_event个事件
        positions = torch.where(inverse_indices == mid_event)[0] #是所有值维mid_event的序列
        # 获取最后一个位置
        last_position = positions[-1]
        # 前半部分事件和每个事件的长度
        event_feats_first = video_feats[0][:last_position + 1]
        #获取前半部分一共有多少个事件,包含mid_event
        event_length_first=event_duration[:mid_event + 1]
        abstract_feat[:mid_event+1]=self.abstract_encoder1(event_feats_first,event_length_first)
        
        event_feats_last = video_feats[0][last_position + 1:] #视频帧是用帧索引来切片
        #获取前半部分一共有多少个事件,包含mid_event
        event_length_last=event_duration[mid_event + 1:]  #事件长度使用事件偏好来切片
        abstract_feat[mid_event+1:]=self.abstract_encoder1(event_feats_last,event_length_last)
        # 记录结束时间
        
        mem_alloc_after = torch.cuda.memory_allocated()
        mem_reserved_after = torch.cuda.memory_reserved()
        # 计算内存增量（以字节为单位）
        mem_alloc_diff = mem_alloc_after - mem_alloc_before
        mem_reserved_diff = mem_reserved_after - mem_reserved_before
        print(f'Allocated memory difference: {mem_alloc_diff / (1024**2):.2f} MB')
        print(f'Reserved memory difference: {mem_reserved_diff / (1024**2):.2f} MB')
        
        ab_et=time.time()
        # print(f"abstract_time: {ab_et-ab_st}")
        event_duration=event_duration/frame_rate  #由原来的帧数,转换为秒数
        bbox_bias = self.regressor(abstract_feat,query_feats,mode="train")
        
        pred_timestamps = event_timestamps/frame_rate + bbox_bias * event_duration.unsqueeze(-1)
            
        
        
        # anchor_event_et=time.time()
        # print(f"anchor_event_time: {anchor_event_et-anchor_event_st}")
        final_event_anchor_feats=event_avg_feats
        
        # """ 根据查询事件相似度进行共享事件筛选 """
        # #region
        # # F.normalize是将张量进行L2范数归一化，p=2表示L2范数，计算方法是每个元素的平方和再开方；dim=-1表示对最后一个维度进行操作
        # # F.normalize在最后一个特征维度上对张量进行归一化，能够便于计算张量之间的相似度，规避量级的影响
        # # final_event_anchor_feats_norm = F.normalize(final_event_anchor_feats, p=2, dim=-1)  # (num_anchors, hidden_dim)
        # # query_feats_norm = F.normalize(query_feats, p=2, dim=-1)  # (batch_query, hidden_dim)
        # # #使用余弦相似度计算其值能达到在【-1，1】之间，更好去判断相似的程度 ，维度为【batch_query,event anchor】
        # # #此处得到的是所有事件和查询之间的相似度，进一步才根据相似度选择出需要学习的事件
        # # similarity_matrix = torch.nn.functional.cosine_similarity(query_feats_norm.unsqueeze(1), final_event_anchor_feats_norm.unsqueeze(0), dim=-1)
        
        # # # todo: baseline的时候先不进行事件之间的聚集，再后面的版本中再进行聚集
        # # #根据相似度阈值选择出对于每个查询需要在视频中使用学习的事件
        # # event_cluster_threshold = 0.03
        # # selected_columns = similarity_matrix > event_cluster_threshold #得到的依然是索引，不是真实事件ID，表示的是第几个事件

        # # # 结果是一个列表，每个元素是一个张量，包含每一行中满足条件的列索引，此处还是需要每个事件对应的时间区间,是有重复值的，不同的样本查询
        # # selected_indices = [torch.nonzero(row).squeeze(1) for row in selected_columns]

        # # #在训练中的计算是共享事件，所有查询使用的是同一个事件集合
        # # combined_indices = torch.cat(selected_indices)
        # # hit_indices = torch.unique(combined_indices)  #是一个一维张量
        # #endregion
        
        
        # """ 使用classfication进行事件命中筛选 """
        # #得到的是 num_query,num_event维度结果，表示每个查询对于每个事件的命中率
        # # event_cluster_threshold=0.55
        
        # class_st=time.time()
        
        classficate_result=self.classifier(query_feats,final_event_anchor_feats) #计算损失的时候没有问题，只是计算hit_ratio的时候不能按照共享来算
        
        # class_et=time.time()
        
        # # print(f"classification_time: {class_et-class_st}")
        top_k=100
        #topk_indices:num_query,top_k  每个查询选择前100个
        _, topk_indices = torch.topk(classficate_result, top_k, dim=1)
        # #在 num_query,top_k个数中进行去重选择，得到的结果将会更少，如果阈值方式的话很难控制
        
        # hit_indices = torch.unique(topk_indices)  #是一个一维张量
        
        
        # """ 抽象化不同长度的事件特征 """
        # #region
        # #这部分时间开销还是比较大，从原来的0.9s降低到0.25s，结果是一样的
        # # 1. 提取每个事件的帧特征，使用mask方法会少较多的判断，相比于inverse_indices==i(只不过这个i是列表，所有的元素)
        # # mask = torch.isin(inverse_indices, hit_indices) #inverse_indices中的元素是否在hit_indices中
        # # selected_feats = video_feats[0][mask]  # 选出的帧特征，是hit_indices中所有命中的事件帧
        # # selected_event_indices = inverse_indices[mask]  # 将选择出来的selected_feats中每一帧对应的事件索引也得到
        # # aggregated_feats = []
        # # for event_idx in hit_indices: #是按照hit_indices中的顺序计算的
        # #     event_feats = selected_feats[selected_event_indices == event_idx]
        # #     aggregated_feat = self.abstract_encoder(event_feats)
        # #     aggregated_feats.append(aggregated_feat)
        # # #abstract_events的行关系是和hit_indices中元素出现顺序一一对应
        # # abstract_events = torch.stack(aggregated_feats)  #stack的默认维度是0 
        # #endregion
        
        # # ab_st=time.time()
        # # # 所有事件的抽象表示
        # # abstract_events=torch.zeros(unique_events.size(0),15,512,device=query_feats.device)
        # # for i in range(unique_events.size(0)):  #抽象事件也是从0号事件开始计算的
        # #     event_feat = video_feats[0][inverse_indices==i] #inverse_indices是视频共享的
        # #     abstract_events[i] = self.abstract_encoder(event_feat)
        # # ab_et=time.time()
        
        # # mask_st=time.time()
        # # abstract_events=torch.zeros(unique_events.size(0),15,512,device=query_feats.device)
        # # #如此弄一个mask就不用像上面一样，每次都要去索引判断一下了
        # # mask = (inverse_indices.unsqueeze(0) == torch.arange(unique_events.size(0), device=inverse_indices.device).unsqueeze(1))
        # # for i in range(unique_events.size(0)):
        # #     # 使用掩码提取每个事件对应的帧特征
        # #     event_feat = video_feats[0][mask[i]]
        # #     abstract_events[i] = self.abstract_encoder(event_feat)
            
        # # abstract_batch=200
        # # num_iter=(unique_events.size(0)+abstract_batch-1)//abstract_batch
        
        # #region
        # #用批量化的方式对事件进行抽象化编码
        # # 按帧数对事件进行排序
        # # 计算每个事件的长度，中的每一个元素是事件编号和该事件对应的帧数长度
        
         
        # batch_ab_st=time.time()
        # event_length_mask_st=time.time()
        # event_length_st=time.time()
        # #耗时0.06比较严重，其实含义就是事件的长度duration的概念，这样时间变短很多
        # event_lengths1=event_timestamps[:,1]-event_timestamps[:,0]  #这个比下面少1帧，索引就是对应的事件编号
        # # event_lengths = [(i, (inverse_indices == i).sum().item()) for i in range(unique_events.size(0))]
        # event_length_et=time.time()
        # # print(f"event_length_time: {event_length_et-event_length_st}")  
        
        # so_st=time.time()
        # # event_lengths.sort(key=lambda x: x[1])  # 按帧数排序，从小到大进行事件按帧数进行排序，这个时间开销也不大啊，0.0002
        # sorted_event_length,indice=event_lengths1.sort()  # 按帧数排序，从小到大进行事件按帧数进行排序，这个时间开销也不大啊，0.0002
        # event_length1 = torch.quantile(sorted_event_length.float(), 0.5)
        # max_length = int(event_length1.item())
        
        # so_et=time.time()
        # # print(f"sort_time: {so_et-so_st}")
        
        
        # batch_size = 1000  # 假设每次处理 10 个事件
        # abstract_events = torch.zeros(unique_events.size(0), 15, 512, device=query_feats.device)
        # mask_st=time.time()
        # #耗时非常小
        # mask = (inverse_indices.unsqueeze(0) == torch.arange(unique_events.size(0), device=inverse_indices.device).unsqueeze(1))
        # mask_et=time.time()
        # # print(f"mask_time: {mask_et-mask_st}")
         
        
        # event_length_mask_et=time.time()
        
        # # print("event_length_mask_time: ",event_length_mask_et-event_length_mask_st)
        # total_indice_time=0
        # total_pad_time=0
        # total_feat_time=0
        # total_allo_time=0
        # # 应该是按照事件长度来划分比较好，比如帧数都小于20帧的一起处理，20-40帧的一起处理
        # # 因为在最后的batch_size的时候可能因为少数的几个长度较大的，导致pad的太多，消耗变大
        
        
        # for start in range(0, len(sorted_event_length), batch_size):
        #     batch_indice_st=time.time()
            
        #     batch_indices = [idx for idx in indice[start:start + batch_size]]  #这个事件0.002其实还好
        #     batch_indice_et=time.time()
        #     print(f"batch_indice_time: {batch_indice_et-batch_indice_st}")
        #     batch_indices = indice[start:start + batch_size]
            
        #     batch_event_st=time.time()
        #     # batch_event_feats = [video_feats[0][inverse_indices == i] for i in batch_indices]
        #     batch_event_feats = [video_feats[0][mask[i]] for i in batch_indices] #这个时间 0.07开销较大
            
        #     batch_event_et=time.time()
        #     print(f"batch_event_time: {batch_event_et-batch_event_st}")
            
        #     # 计算每批次的最大帧数
        #     max_frames = max(feat.size(0) for feat in batch_event_feats)
        #     indice_et=time.time()
            
        #     # 创建一个适当大小的张量
        #     batch_padded_feats = torch.zeros(len(batch_event_feats), max_frames, 512, device=query_feats.device)
            
        #     pad_st=time.time()
        #     for i, feat in enumerate(batch_event_feats):
        #         batch_padded_feats[i, :feat.size(0)] = feat
        #     pad_et=time.time()
            
        #     # 通过 abstract_encoder 批量处理
        #     feat_st=time.time()
        #     encoded_feats = self.abstract_encoder(batch_padded_feats)
        #     feat_et=time.time()
            
        #     # for i, idx in enumerate(batch_indices):
        #     #     abstract_events[idx] = encoded_feats[i]
            
        #     allo_st=time.time()
        #     abstract_events[batch_indices] = encoded_feats
        #     allo_et=time.time()
        #     total_indice_time+=indice_et-indice_st
        #     total_pad_time+=pad_et-pad_st
        #     total_feat_time+=feat_et-feat_st
        #     total_allo_time+=allo_et-allo_st
        #     print(f"indice_time:{indice_et-indice_st};pad_time:{pad_et-pad_st};feat_time:{feat_et-feat_st};allo_time:{allo_et-allo_st}")
        #     ml=1*2

        # batch_ab_et=time.time()
        
        # print(f"total_indice_time:{total_indice_time};total_pad_time:{total_pad_time};total_feat_time:{total_feat_time};total_allo_time:{total_allo_time}")
        
            
        # mask_et=time.time()
        # #region
        # #选择事件
        # # select_abstract_feature=abstract_events[topk_indices]  #topk_indices和0，1这些进行了映射
        # # select_event_timestamps=event_timestamps[topk_indices]

        # # """不再是所有的事件，索引不再对应，有一个压缩映射效果"""
        # # #下面这样的索引方式得到的结果和hit_indices的顺序是一致的，将hit_indices的绝对索引映射到0-len(hit_indices)中压缩
        # # #其实这个映射就是hit_indices的索引
        # # # select_event_timestamps=event_timestamps[hit_indices]  #得到选择的事件的时间戳，用于后面将边界偏移转换为绝对时间
        # # select_duration=select_event_timestamps[:,1]-select_event_timestamps[:,0]  #得到选择的事件的持续时间
        
        # # #下面四个是一一对应的  abstract_events: num_event,15,512
        # # #bbox_bias: num_query, num_event, 2,对于event顺序是和hit_indices一致的【是每一个事件一个结果】
        # # bbox_bias = self.regressor(select_abstract_feature, query_feats,mode="train") #结合context-based和content-based的特征进行bbox回归
        # # select_event_timestamps=select_event_timestamps.to(bbox_bias.device)
        # # select_duration=select_duration.to(bbox_bias.device)
        # # #下面这个广播机制没有问题
        # # pred_timestamps = select_event_timestamps + bbox_bias * select_duration.unsqueeze(1)
        # # #要去计算原本聚类得到的事件到底哪些和真实查询的标注存在相交，用这部分的事件得到的预测值进行损失的计算才对(不然选择无关的片段强行将时间对齐，学习到的内容将有问题)
        # # #最终可能就少数几个事件和真实标注的区间相关，没有多尺度
        # #endregion
        
        # """ 选择损失计算的目标事件 """
        # event_overlaps_pred=list()
        # #要将timestamps由原本的s为单位转换为帧才能确定对应的事件，round得到的是取整后的浮点数，还不是integer形式
        frame_timestamps=torch.round(timestamps*5).int() #真实标注的帧区间，有的秒数可能不足1s，但是帧数肯定大于一帧，取round应该没啥问题
        
        # #classifier target & regressor target 
        # #得到每个查询对于每个事件的命中情况，0-1二分类，初始化全0，当命中才设置为1 
        
        hit_ratio=0
        classifier_target=torch.zeros((query_feats.size(0),len(unique_events)),device=query_feats.device)
        # # event_overlaps_pred,hit_ratio,classifier_target =get_targets(pred_timestamps,hit_indices,classifier_target,frame_timestamps,inverse_indices)
        # #下面的一个region是未封装的代码
        
        # query_mask=torch.zeros(query_feats.size(0),device=query_feats.device) #表示每个查询到底命中没有要不要参与后面的回归损失计算
        
        # #region
        # #对于选择事件
        # # for query_index,frame_timestamp in enumerate(frame_timestamps): #对于每一个查询来计算与事件的命中关系
        # #     overlap_events=torch.unique(inverse_indices[frame_timestamp[0]:frame_timestamp[1]+1]) #这个overlap_events是0开始的
        # #     classifier_target[query_index][overlap_events]=1  #针对每个查询的命中程度
        # #     overlap_event_predict=[] #针对每个查询都得到的一个目标损失预测
        # #     flag=0  #如果该query一个事件都没命中就是0，只要有一个命中就是1，就可以参与后面的回归损失计算
        # #     for overlap_event in overlap_events: 
        # #         #hit_indices和overlap_events都是绝对的事件编号 
        # #         #topk_indices经过和abstract_feature索引之后进行了映射
        # #         if overlap_event in topk_indices[query_index]:  #overlap_event是一个一个进行遍历计算的
        # #             #选择得到的事件、预测结果的索引和hit_indices存在压缩对应的关系，位序编号保持一致
        # #             #原本在hit_indices中第1个元素为6(表示第6个事件)，在pred_timestamps中将会变为第一个元素值，对应的是所有事件的第6个
        # #             #因此应该找hit_indices中值(事件)的索引，在选择后的事件中使用该索引得到该事件的预测值
        # #             # overlap_index = (hit_indices == overlap_event).nonzero(as_tuple=True)[0]
        # #             overlap_index=topk_indices[query_index].tolist().index(overlap_event)
        # #             overlap_event_predict.append(pred_timestamps[query_index][overlap_index].squeeze())#得到的是绝对的时间，如果不使用squeeze会多一个维度
        # #             flag=1
        # #         else: #这里这个逻辑还是比较奇怪，如果预测命中事件没有命中的话
        # #             # 不去计算就行
        # #             #额外设置的这个值没有梯度无法更新，只作为一个标记
        # #             overlap_event_predict.append(torch.tensor([-1,-1],device=query_feats.device))   #额外设置的这个值没有梯度无法更新

        # #         # if overlap_event in topk_indices[query_index]: #应该去比较每个query的选择，而不是共享的选择
        # #         #     flag=1
            
        # #     if flag==1: #表示命中
        # #         hit_ratio+=1
        # #     else:
        # #         query_mask[query_index]=1  #query_mask值为1，表示未命中
            
        # #     event_overlaps_pred.append(torch.stack(overlap_event_predict)) 
        # #endregion

        # # 针对所有事件进行计算回归
        # # 如此的话每个查询必命中，因为是对所有的事件回归的
        # # bbox_bias = self.regressor(abstract_events, query_feats,mode="train")
        # event_duration=event_timestamps[:,1]-event_timestamps[:,0]
        # #event_timestamps : num_event,2  ;bbox_bias:num_query,num_event,2 ;event_duration: num_event
        # # pred_timestamps = event_timestamps + bbox_bias * event_duration.unsqueeze(-1)
        
        event_overlaps_pred=list()
        
        # ta_st=time.time()
        for query_index,frame_timestamp in enumerate(frame_timestamps):
            overlap_event_predict=[]
            # 拿到的是事件的索引：第几个事件,真实标注中的哪些事件
            overlap_events=torch.unique(inverse_indices[frame_timestamp[0]:frame_timestamp[1]+1])
            classifier_target[query_index][overlap_events]=1  #针对每个查询的命中程度
            
            flag=0
            for overlap_event in overlap_events:
                if overlap_event in topk_indices[query_index]:
                    flag=1
                    
            if flag==1:
                hit_ratio+=1
                
            #把对应的标注时间拿出来
            overlap_timestamps=pred_timestamps[query_index][overlap_events] # 针对每个查询，在指定事件上的预测
            # overlap_event_predict.append(overlap_timestamps)
            event_overlaps_pred.append(overlap_timestamps) 
            
        # ta_et=time.time()
        

        # # print("hit_ratio: ",hit_ratio/query_feats.size(0))
        # # event_overlaps_pred=list()
        # # #要将timestamps由原本的s为单位转换为帧才能确定对应的事件，round得到的是取整后的浮点数，还不是integer形式
        # # frame_timestamps=torch.round(timestamps*5).int() #真实标注的帧区间，有的秒数可能不足1s，但是帧数肯定大于一帧，取round应该没啥问题
        
        # #region
        # # #对每个查询而言
        # # for query_index,frame_timestamp in enumerate(frame_timestamps):
        # #     #计算查询真实标注区间内对应的事件
        # #     overlap_events=torch.unique(inverse_indices[frame_timestamp[0]:frame_timestamp[1]+1])  #得到真实标注区间对应的事件索引
        # #     #所有的事件和查询的标注进行相交比较，大于选择的事件数目
        # #     overlap_event_predict=[] #针对每个查询都得到的一个目标损失预测
        # #     for overlap_event in overlap_events: 
        # #         #hit_indices和overlap_events都是绝对的事件编号 
        # #         if overlap_event in hit_indices:  #overlap_event是一个一个进行遍历计算的
        # #             #选择得到的事件、预测结果的索引和hit_indices存在压缩对应的关系，位序编号保持一致
        # #             #原本在hit_indices中第1个元素为6(表示第6个事件)，在pred_timestamps中将会变为第一个元素值，对应的是所有事件的第6个
        # #             #因此应该找hit_indices中值(事件)的索引，在选择后的事件中使用该索引得到该事件的预测值
        # #             overlap_index = (torch.tensor(hit_indices).to(overlap_event.device) == overlap_event).nonzero(as_tuple=True)[0]
        # #             overlap_event_predict.append(pred_timestamps[query_index][overlap_index].squeeze())#得到的是绝对的时间，如果不使用squeeze会多一个维度
        # #         else: 
        # #             #region
        # #             #会存在真实标注的事件没有被选择的情况，这种情况另外计算赋值其他,强制给一个损失还算比较大
        # #             #会存在有的查询一个真实的标注区间都没有检测到
        # #             #应该弄一个是模型输出的东西，具有梯度的值才行，不然无法更新，不能直接凭空加一个东西
        # #             #将某个值通过计算变成0，0 
        # #             #overlap_event_predict.append(torch.tensor([0,0],device=bbox_bias.device))   #额外设置的这个值没有梯度无法更新
        # #             #endregion
        # #             overlap_event_predict.append(torch.sigmoid(pred_timestamps[query_index][0].squeeze()))   #额外设置的这个值没有梯度无法更新
        # #     event_overlaps_pred.append(torch.stack(overlap_event_predict)) 
        # #endregion
        
        
        
        # #event_overlaps_pred 列表，长度和query_mask的维度一样，表示每一行是否命中，是否需要进行后面的计算
        loss_dict=self.loss(event_overlaps_pred,timestamps,classficate_result,classifier_target)  #timestamps是真实绝对的时间戳
        train_et=time.time()
        # print(f"training_time:{train_et-train_st}")
        # print(f"abstract_time: {ab_et-ab_st}, target_time: {ta_et-ta_st},top_time: {top_et-top_st},train_time: {train_et-train_st},maks_time: {mask_et-mask_st}")   
        return loss_dict,hit_ratio/query_feats.size(0)
        # return 0,0

    def forward_test(self,
                     query_feats=None,
                     video_feats=None,
                     video_events=None,
                     timestamps=None,
                     **kwargs):
        """
        Args:

        Returns:                                    
        """        
        video_events=video_events.squeeze() #先考虑bsz为1，不然unique得到的inverse_indices会有问题
        unique_events, inverse_indices = torch.unique(video_events, return_inverse=True) #也可以直接用这个来作为新的事件表示了，统一移动了
        
        hidden_dim=video_feats.size(-1)
        event_avg_feats = torch.zeros((unique_events.size(0), hidden_dim))
        
        """ 计算每个事件的平均特征 """
        event_timestamps=list()
        #对平均事件和事件的时间都是从0开始增加的
        for i in range(unique_events.size(0)):
            frame_rate=5
            #得到每个事件区间的Anchor特征
            event_avg_feats[i] = video_feats[0][inverse_indices == i].mean(dim=0)
            #得到每个事件的开始和结束时间
            event_start_index = (inverse_indices == i).nonzero(as_tuple=True)[0][0]
            event_end_index = (inverse_indices == i).nonzero(as_tuple=True)[0][-1]
            event_start_time=event_start_index
            event_end_time=event_end_index
            event_timestamps.append(torch.tensor([event_start_time,event_end_time]))
            
        event_timestamps=torch.stack(event_timestamps,dim=0)  #得到tensor表示的时间戳，此处都是所有的事件
        event_timestamps=event_timestamps.to(video_feats.device)
        event_duration=event_timestamps[:,1]-event_timestamps[:,0]+1
        
        final_event_anchor_feats=event_avg_feats
        final_event_anchor_feats=final_event_anchor_feats.to(query_feats.device)
        
        #region
        # 将结果添加到列表中
        # event_avg_feats_list.append(event_avg_feats)
        # # 这里最终的输出形状将为 (batch_size, num_events, hidden_dim)，在操作中bsz值为1 ,final_event的维度和unique_events的维度相关
        # final_event_anchor_feats = torch.stack(event_avg_feats_list)
        # final_event_anchor_feats=final_event_anchor_feats.to(query_feats.device)
        # final_event_anchor_feats = final_event_anchor_feats.squeeze(0) 
        #endregion
        # final_event_anchor_feats_norm = F.normalize(final_event_anchor_feats, p=2, dim=-1)  # (num_anchors, hidden_dim)
        
        #query层面小batch的计算
        #需要每个查询选择的事件抽象表示，还需要其对应的时间标注，因为预测得到的是边界的偏移，要计算IOU需要绝对时间
        batch_size = self.cfg.TEST.BATCH_SIZE
        batch_size = 7
        query_num = len(query_feats)
        num_batches = math.ceil(query_num / batch_size)
        all_query_pred_bbox_bias,all_hit_timestamps = list(),list()
        
        # 计算所有事件的抽象特征
        #region
        #计算出所有事件的抽线特征，最后不同查询再选即可
        #这段代码不能放到bid循环中去了，因为是针对事件本身的，不是针对查询，放进去会对每个小query batch又重复计算
        # abstract_features=torch.zeros(unique_events.size(0),15,512,device=query_feats.device)
        # for i in range(unique_events.size(0)):  #抽象事件也是从0号事件开始计算的
        #     event_feat = video_feats[0][inverse_indices==i] #inverse_indices是视频共享的
        #     abstract_features[i] = self.abstract_encoder(event_feat)
        #endregion
        
        abstract_feat=torch.zeros(unique_events.size(0),15,512,device=query_feats.device)
        mid_frame=video_feats[0].size(0)//2
        mid_event=inverse_indices[mid_frame]  #事件是从0开始的,因此一共包含1+mid_event个事件
        positions = torch.where(inverse_indices == mid_event)[0]
        # 获取最后一个位置
        last_position = positions[-1]
        # 前半部分事件和每个事件的长度
        event_feats_first = video_feats[0][:last_position + 1]
        #获取前半部分一共有多少个事件,包含mid_event
        event_length_first=event_duration[:mid_event + 1]
        abstract_feat[:mid_event+1]=self.abstract_encoder1(event_feats_first,event_length_first)
        
        event_feats_last = video_feats[0][last_position + 1:]
        #获取前半部分一共有多少个事件,包含mid_event
        event_length_last=event_duration[mid_event + 1:]
        abstract_feat[mid_event+1:]=self.abstract_encoder1(event_feats_last,event_length_last)
        
        
        hit_ratio=0  #最后算出来是所有的样本
        for bid in range(num_batches): #小query_batch
            query_feats_batch = query_feats[bid * batch_size: (bid + 1) * batch_size]
            timestamp_batch = timestamps[bid * batch_size: (bid + 1) * batch_size]
            frame_timestamps=torch.round(timestamp_batch*5).int()
                
            classficate_result=self.classifier(query_feats_batch,final_event_anchor_feats)
            top_k=100
            #topk_indices:num_query,top_k  这个indices本身就是从大到小来的
            _, top_indices = torch.topk(classficate_result, top_k, dim=1)
            
            
            for query_index,frame_timestamp in enumerate(frame_timestamps): #对于每一个查询来计算与事件的命中关系
                overlap_events=torch.unique(inverse_indices[frame_timestamp[0]:frame_timestamp[1]+1]) #这个overlap_events是0开始的
                flag=0
                for overlap_event in overlap_events:
                    if overlap_event in top_indices[query_index]:
                        flag=1
                if flag==1:
                    hit_ratio+=1
                    
                
            
            #在 num_query,top_k个数中进行去重选择，得到的结果将会更少，如果阈值方式的话很难控制
            # hit_indices = torch.unique(topk_indices)  #是一个一维张量  不再需要共享了，直接每个查询自己的事件就能算了
            
            #region
            #原本直接基于相似度阈值筛选的方法
            # query_feats_norm = F.normalize(query_feats_batch, p=2, dim=-1)  # (batch_query, hidden_dim)
            # similarity_matrix = torch.nn.functional.cosine_similarity(query_feats_norm.unsqueeze(1), final_event_anchor_feats_norm.unsqueeze(0), dim=-1)
            # # 针对每一个查询，选择独立的topk个事件
            # _, top_indices = torch.topk(similarity_matrix, self.stage2_topk, dim=1)
            #endregion
            
            #region
            # abstract_features=torch.zeros((query_feats_batch.size(0),self.stage2_topk,15,512),device=query_feats_batch.device)
            #abstract_features的维度和原项目中ctn_feats的维度一致
            #针对每个事件的时间戳(7,100,2)
            #这样写会对同一个事件因为不同的查询都用到了，多次计算，
            #这么写比较耗时,需要花1.6s
            # st_time=time.time()
            # batch_hit_timestamps=torch.zeros((query_feats_batch.size(0),self.stage2_topk,2),device=query_feats_batch.device)
            # for query_index,indices in enumerate(top_indices):
            #     batch_hit_timestamps[query_index]=event_timestamps[indices]
            #     for indice_index,indice in enumerate(indices):
            #         #对于每个event_feat的长度不是一定的，有的事件可能只有一帧，有的事件可能有很多帧
            #         event_feat = video_feats[0][inverse_indices==indice]
            #         abstract_features[query_index][indice_index] = self.abstract_encoder(event_feat)
            # et_time=time.time()
            # print("time:",et_time-st_time)
            
            # stb_time=time.time()
            # 初始化 batch_hit_timestamps
            # 高级索引，top_indices是二维矩阵(100,100)，event_timestamps是二维矩阵(n,2)
            # 在进行索引的时候会在top_indices的第一个维度表示批量，第二个维度100去匹配event_timestamps中的n，剩余的2是得到的结果
            #endregion
            #top_indices是一个二维矩阵，第一个维度是batch_size，第二个维度是topk
            #在索引的时候，第一个维度不变，用第二个维度去event_timestamps里面进行索引
            batch_hit_timestamps = event_timestamps[top_indices]  #event_timestamps和abstract_features的索引是一一对应关系
            
            #region
            # 关键是event_feat长度不一定，有的事件可能只有一帧，有的事件可能有很多帧，不好批量处理
            # 先将所有查询使用的事件并集起来，一次for循环执行，最后再选？
            
            #计算出所有事件的抽线特征，最后不同查询再选即可
            # abstract_features=torch.zeros(unique_events.size(0),15,512,device=query_feats_batch.device)
            # for i in range(unique_events.size(0)):
            #     event_feat = video_feats[0][inverse_indices==i]
            #     abstract_features[i] = self.abstract_encoder(event_feat)
            #endregion
            
            #在训练的时候使用的是多个查询之间共享的事件，但是在测试的时候，每个查询是自己的事件
            select_abstract_feature=abstract_feat[top_indices]  #top_indices对每个查询事件选择不一样，因此得到的时间区间也不一样

            #select_abstract_feature : num_query,100,15,512 在训练的时候，(num_events,15,512)
            
            #得到的是batch_size,100,2；但最终需要得到总query,100,2  
            bbox_bias = self.regressor(select_abstract_feature, query_feats_batch,mode="test")
            all_query_pred_bbox_bias.append(bbox_bias) #每个查询关于自己选择的事件预测偏移:num_query,100,2
            all_hit_timestamps.append(batch_hit_timestamps) #每个查询选择的事件的时间戳:num_query,100,2
            
            
            
        
        all_query_pred_bbox_bias = torch.cat(all_query_pred_bbox_bias, dim=0)
        all_hit_timestamps=torch.cat(all_hit_timestamps,dim=0)/frame_rate
        all_hit_duration=all_hit_timestamps[:,:,1]-all_hit_timestamps[:,:,0]  #num_query,100
        pred_timestamps = all_hit_timestamps + all_query_pred_bbox_bias * all_hit_duration.unsqueeze(-1) #这个广播机制也没错

        # print("Test hit_ratio: ",hit_ratio/query_feats.size(0))
        return pred_timestamps,hit_ratio/query_feats.size(0)
        # return 0,hit_ratio/query_feats.size(0)  #一个视频下所有查询的命中情况

    def loss(self, 
             bbox_bias, #经过与真实片段存在相交的事件的预测值，列表数据一共timestamps.size(0)个元素
             timestamps,
             classficate_result,
             classifier_target
             ):
        
        iou_loss =0.0
        iou_st=time.time()
        # iou_loss = self.reg_loss(bbox_bias, timestamps,query_mask)
        iou_loss = self.reg_loss(bbox_bias, timestamps)
        iou_et=time.time()
        
        cl_st=time.time()
        classficate_loss=self.classify_loss(classficate_result,classifier_target)  #分类的损失不用变
        cl_et=time.time()

        #命中与否对于性能的影响更大，因此权重设置的更大，regress是20
        # total_loss = self.cfg.LOSS.REGRESS.WEIGHT * iou_loss+classficate_loss *30
        # if 
        total_loss =  iou_loss+classficate_loss *100.0  #把量级调整一下

        # print(f"iou_time:{iou_et-iou_st}.class_time:{cl_et-cl_st}")
        loss_dict = {
            "reg_loss": iou_loss,
            "reg_loss_W": iou_loss*0.1,
            "classficate_loss":classficate_loss,
            "classficate_loss_W":classficate_loss*5.0,
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
    
    
    