import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import fetch_feats_by_index



class Q2VRankerStage1(nn.Module):

    def __init__(self, nlevel, hidden_dim): #nlevel就是尺度数
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.nlevel = nlevel

    def forward(self, ctx_feats, qfeat):
        qfeat = self.fc(qfeat)  #bsz 每一个query与同一个视频进行计算，重复利用
        qv_ctx_scores = list()
        for i in range(self.nlevel):
            #F.normalize(input, p=2, dim=1)表示对input的第1维度进行L2（p=2)归一化
            #爱因斯坦求和（einsum），"bld,bd->bl"表示张量间的乘法操作 b表示batch_size, l表示序列长度，d表示维度
            #计算的是 ctx_feats[i]中每个时间步的特征向量与qfeat特征向量之间的余弦相似度，得到的分数维度是(batch_size, frames_len)
            score = torch.einsum("bld,bd->bl", 
                    F.normalize(ctx_feats[i], p=2, dim=2), F.normalize(qfeat, p=2, dim=1))
            qv_ctx_scores.append(score)

        return qv_ctx_scores  #计算每一个anchor和query之间的得分，其中query有多个


class V2QRankerStage1(nn.Module):

    def __init__(self, nlevel, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.nlevel = nlevel

    def forward(self, ctx_feats, qfeat):
        vq_ctx_scores = list()
        for i in range(self.nlevel):
            #与前面的区别在于这里是先对ctx_feats[i]进行线性变换，然后再进行归一化，Q2VRankerStage1是对query进行线性变换再进行计算
            score = torch.einsum("bld,bd->bl", 
                    F.normalize(self.fc(ctx_feats[i]), p=2, dim=2), F.normalize(qfeat, p=2, dim=1))
            vq_ctx_scores.append(score)
        
        return vq_ctx_scores


class Q2VRankerStage2(nn.Module):

    def __init__(self, nlevel, hidden_dim, snippet_length=10, pool='mean'):
        super().__init__()
        self.nlevel = nlevel
        self.base_snippet_length = snippet_length
        self.qfc = nn.Linear(hidden_dim, hidden_dim)
        self.encoder = V2VAttention(hidden_dim)
        self.pool = pool

    def forward(self, vfeats, qfeat, hit_indices, qv_ctx_scores):
        """_summary_

        Args:
            vfeats (_torch_): _表示视频特征(1,frames,hidden_dim)_
            qfeat (_torch_): _(query_batch,hidden_Dim)_
            hit_indices (_list_): 四个元素，每个元素表示该尺度下选择了哪些anchor
            qv_ctx_scores (_type_): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            qv_merge_scores: list,(query_batch,selected_anchors)每个元素表示在当前尺度下，每个选择到的anchor的得分 融合context-based和content-based的分数,
            ctn_feats还是列表，内部每个元素维度为(selected_anchors, snippet_length, D)
        """
        qfeat = self.qfc(qfeat)

        qv_ctn_scores = list()
        qv_merge_scores = list()

        _, L, D = vfeats.size()  #没有进行维度转换
        ctn_feats = list()
        for i in range(self.nlevel):
            snippet_length = self.base_snippet_length * 2**i  #和dataloader中的处理方式一样
            assert L // snippet_length == qv_ctx_scores[i].size(1), \
                "{}, {}, {}, {}".format(i, L, snippet_length, qv_ctx_scores[i].size())
            
            #此部分还是视频的特征，没有content-based,也要将视频划分为不同的区间
            ctn_feat = vfeats.view(L//snippet_length, snippet_length, D).detach()
            if self.training:
                qv_ctx_score = torch.index_select(qv_ctx_scores[i], 1, hit_indices[i])
                ctn_feat = torch.index_select(ctn_feat, 0, hit_indices[i])  #0,表示没有bsz维度，将有相交部分的内容选出来，得到的还是tensor张量只不过行数变少
                #多个anchor一起编码 (selected_anchors, snippet_length, D)
                #encoder经检验不会改变输入数据的维度，即输出仍为(selected_anchors, snippet_length, D)
                ctn_feat = self.encoder(ctn_feat, torch.ones(ctn_feat.size()[:2], device=ctn_feat.device))
                ctn_feat = ctn_feat.unsqueeze(0)  #构造出一个batch_size维度 (1,selected_anchors, snippet_length, D)
                
            else:
                qv_ctx_score = fetch_feats_by_index(qv_ctx_scores[i], hit_indices[i])
                B, K = hit_indices[i].shape
                ctn_feat = fetch_feats_by_index(ctn_feat.unsqueeze(0).repeat(B, 1, 1, 1), hit_indices[i]).view(B*K, snippet_length, D)
                ctn_feat = self.encoder(ctn_feat, torch.ones(ctn_feat.size()[:2], device=ctn_feat.device))
                ctn_feat = ctn_feat.view(B, K, snippet_length, D)
            
            ctn_feats.append(ctn_feat)  #每个尺度下都添加到ctn中，列表中四个元素，每个元素(1,selected_anchors, snippet_length, D)
            
            # 下面计算content-based与文本的分数
            qv_ctn_score = torch.einsum("bkld,bd->bkl", 
                           F.normalize(ctn_feat, p=2, dim=3), F.normalize(qfeat, p=2, dim=1))
            if self.pool == "mean":
                qv_ctn_score = torch.mean(qv_ctn_score, dim=2)  #这个部分去掉snippet_length维度
            elif self.pool == "max":
                qv_ctn_score, _ = torch.max(qv_ctn_score, dim=2)
            else:
                raise NotImplementedError
            qv_ctn_scores.append(qv_ctn_score)
            qv_merge_scores.append(qv_ctx_score + qv_ctn_score)
        
        #分数为列表，四个元素，表示四个尺度，每个元素的维度为(query_batch_size, selected_anchors)
        return qv_merge_scores, qv_ctn_scores, ctn_feats  #ctn_feats还是列表，内部每个元素维度为(selected_anchors, snippet_length, D)


class V2QRankerStage2(nn.Module):

    def __init__(self, nlevel, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.nlevel = nlevel

    def forward(self, ctn_feats, qfeat):
        vq_ctn_scores = list()
        for i in range(self.nlevel):
            score = torch.einsum("bkld,bd->bkl", 
                    F.normalize(self.fc(ctn_feats[i]), p=2, dim=3), F.normalize(qfeat, p=2, dim=1))
            score = torch.mean(score, dim=2)
            vq_ctn_scores.append(score)
        
        return vq_ctn_scores

class V2VAttention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.posemb = PositionEncoding(max_len=400, dim=hidden_dim, dropout=0.0)
        self.encoder = MultiHeadAttention(dim=hidden_dim, n_heads=8, dropout=0.1)
        self.dropout = nn.Dropout(0.0)

    def forward(self, video_feats, video_masks):
        mask = torch.einsum("bm,bn->bmn", video_masks, video_masks).unsqueeze(1)
        residual = video_feats
        video_feats = video_feats + self.posemb(video_feats)
        out = self.encoder(query=video_feats, key=video_feats, value=video_feats, mask=mask)
        video_feats = self.dropout(residual + out) * video_masks.unsqueeze(2).float()
        return video_feats


class BboxRegressor(nn.Module):

    def __init__(self, hidden_dim, enable_stage2=False):
        super().__init__()
        self.fc_ctx = nn.Linear(hidden_dim, hidden_dim)
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        
        if enable_stage2:
            self.fc_ctn = nn.Linear(hidden_dim, hidden_dim)
            self.attn = SelfAttention(hidden_dim)
            self.predictor = nn.Sequential(
                nn.Linear(2*hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2) 
            )
        else:
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2) 
            )
        self.enable_stage2 = enable_stage2

    def forward(self, ctx_feats, ctn_feats, qfeat): 
        """_summary_

        Args:
            ctx_feats (_list_]): 长度为尺度数，每个元素为在该尺度下的(1,selected_anchors,hidden_dim)
            ctn_feats (_list_):  长度为尺度数，每个元素为在该尺度下的(1,selected_anchors,snippet_length,hidden_dim)
            qfeat (_torch_): 文本特征

        Returns:
            _torch_: 维度(query_length, total_num_anchors, 2)
        """
        #传入的ctx,ctn都是列表，列表中的元素才是特征，每个特征的维度保持一致
        qfeat = self.fc_q(qfeat)
        #(1,total_num_anchors,hidden_dim) 所有尺度下anchor数目加起来，因为anchor没有长度，所以可以直接合并
        ctx_feats = torch.cat(ctx_feats, dim=1) 
        #模态交互点积操作 (query_length, total_num_anchors, hidden_dim)
        ctx_fuse_feats = F.relu(self.fc_ctx(ctx_feats)) * F.relu(qfeat.unsqueeze(1))  #在这里query_length应该指的是bsz
        if self.enable_stage2 and ctn_feats:
            ctn_fuse_feats = list()
            #每个元素(1，selected_anchors, snippet_length, D)
            for i in range(len(ctn_feats)):  #长度就是scale的数目,每一个尺度下的ctn_feat维度为(1,selected_anchors,snippet_length,hidden_dim)
                #将query的维度广播，video anchor的维度不变(1,selected_anchors,snippet_length,hidden_dim)
                out = F.relu(self.fc_ctn(ctn_feats[i])) * F.relu(qfeat.unsqueeze(1).unsqueeze(1))
                out = self.attn(out) #在attn中消除snippet_lenghth的原理
                ctn_fuse_feats.append(out) 
                
            #(query_length, total_num_anchors, hidden_dim) 第一个维度是多少个query，第二个维度为每个query和每个anchor的得分
            ctn_fuse_feats = torch.cat(ctn_fuse_feats, dim=1) #因为要cat起来，那应该没有snippet_length这个维度，在attn中已经解决
            fuse_feats = torch.cat([ctx_fuse_feats, ctn_fuse_feats], dim=-1) #hidden_dim拼接
        else:
            fuse_feats = ctx_fuse_feats
        
        #最终结果为 (query_length, total_num_anchors, 2)即对每个anchor都进行了时间边界的预测
        out = self.predictor(fuse_feats)
        return out


class SelfAttention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim//2, 1) #将最后一个hidden_dim转变为1

    def forward(self, x):
        #因为x中包含snippet length维度，会计算得到片段内每一帧与文本查询的相似度，从整体片段为单位来看，需要将维度合并
        att = self.fc2(self.relu(self.fc1(x))).squeeze(3) 
        att = F.softmax(att, dim=2).unsqueeze(3)  #又将最后一个维度加回来
        out = torch.sum(x * att, dim=2) #x * att 广播机制，对att最后一个维度复制到x最后一个维度数
        #通过sum函数将snippet_length维度消除掉，相当于对每一个anchor内部的帧的相似度求和得到一个结果
        return out


class PositionEncoding(nn.Module):

    def __init__(self, max_len, dim, dropout=0.0):
        super(PositionEncoding, self).__init__()

        self.embed = nn.Embedding(max_len, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos_ids = pos_ids.unsqueeze(0).repeat(batch_size, 1)
        pos_emb = self.dropout(self.relu(self.embed(pos_ids)))

        return pos_emb



class MultiHeadAttention(nn.Module):

    def __init__(self, dim, n_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # (N, nh, L, dh)
    
    def forward(self, query, key, value, mask):
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)

        q_trans = self.transpose_for_scores(q)
        k_trans = self.transpose_for_scores(k)  
        v_trans = self.transpose_for_scores(v)  

        att = torch.matmul(q_trans, k_trans.transpose(-1, -2))  # (N, nh, Lq, L)
        att = att / math.sqrt(self.head_dim)
        att = mask_logits(att, mask)
        att = self.softmax(att)
        att = self.dropout(att)

        ctx_v = torch.matmul(att, v_trans)  # (N, nh, Lq, dh)
        ctx_v = ctx_v.permute(0, 2, 1, 3).contiguous()  # (N, Lq, nh, dh)
        shape = ctx_v.size()[:-2] + (self.dim, )
        ctx_v = ctx_v.view(*shape)  # (N, Lq, D)
        return ctx_v


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value
