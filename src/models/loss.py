# Adapted from https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/python/losses_impl.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ApproxNDCGLoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.alpha = cfg.LOSS.TEMPE

    def forward(self, labels, logits, mask):
        if logits is None:
            return 0.0
        labels = labels / self.alpha
        logits = torch.where(mask.bool(), logits, torch.min(logits, dim=1, keepdim=True)[0] - 1e3 * torch.ones_like(logits))
        logits = logits / self.alpha
        ranks = self.approx_ranks(logits)

        loss = 1.0 - self.ndcg(labels, ranks)
        return loss.sum()
    

    def approx_ranks(self, logits):
        r"""Computes approximate ranks given a list of logits.
        Given a list of logits, the rank of an item in the list is one plus the total
        number of items with a larger logit. In other words,
            rank_i = 1 + \sum_{j \neq i} I_{s_j > s_i},
        where "I" is the indicator function. The indicator function can be
        approximated by a generalized sigmoid:
            I_{s_j < s_i} \approx 1/(1 + exp(-(s_j - s_i)/temperature)).
        This function approximates the rank of an item using this sigmoid
        approximation to the indicator function. This technique is at the core
        of "A general approximation framework for direct optimization of
        information retrieval measures" by Qin et al.
        Args:
            logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
            ranking score of the corresponding item.
        Returns:
            A `Tensor` of ranks with the same shape as logits.
        """
        list_size = logits.size(1)
        x = logits.unsqueeze(2).repeat(1, 1, list_size)
        y = logits.unsqueeze(1).repeat(1, list_size, 1)
        pairs = torch.sigmoid(y - x)
        return pairs.sum(dim=-1) + .5


    def ndcg(self, labels, ranks):
        """Computes NDCG from labels and ranks.
        Args:
            labels: A `Tensor` with shape [batch_size, list_size], representing graded
            relevance.
            ranks: A `Tensor` of the same shape as labels, or [1, list_size], or None.
            If ranks=None, we assume the labels are sorted in their rank.
            perm_mat: A `Tensor` with shape [batch_size, list_size, list_size] or None.
            Permutation matrices with rows correpond to the ranks and columns
            correspond to the indices. An argmax over each row gives the index of the
            element at the corresponding rank.
        Returns:
            A `tensor` of NDCG, ApproxNDCG, or ExpectedNDCG of shape [batch_size, 1].
        """
        discounts = 1. / torch.log1p(ranks.float())
        gains = torch.pow(2., labels) - 1.

        dcg = (gains * discounts).sum(1, keepdim=True)
        normalized_dcg = dcg * self.inverse_max_dcg(labels)

        return normalized_dcg

    def inverse_max_dcg(self, labels, 
                        gain_fn=lambda labels: torch.pow(2.0, labels)-1., 
                        rank_discount_fn=lambda rank: 1./torch.log1p(rank),
                        topn=None):
        ideal_sorted_labels = self.sort_by_scores(labels, topn=topn)
        rank = (torch.arange(ideal_sorted_labels.size(1)) + 1).to(labels.device)
        discounted_gain = gain_fn(ideal_sorted_labels) * rank_discount_fn(rank.float())
        discounted_gain = discounted_gain.sum(1, keepdim=True)
        idcg = torch.where(torch.greater(discounted_gain, 0.0), 1./discounted_gain, torch.zeros_like(discounted_gain))

        return idcg

    def sort_by_scores(self, scores, mask=None, topn=None):
        list_size = scores.size(1)
        if topn is None:
            topn = list_size
        topn = min(topn, list_size)
        if mask is not None:
            scores = torch.where(mask.bool(), scores, torch.min(scores))
        sorted_scores, sorted_indices = torch.topk(scores, topn)
        return sorted_scores



class IOULoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.reduce = cfg.LOSS.REGRESS.REDUCE
    
    def forward(self, pred,ground_truth,query_mask):
        #原本项目使用的是绝对的时间戳来计算的损失，不是相对[0,1]
        # pred在训练时是一个列表，每个元素表示该查询的真实标注区间中对应的事件的预测结果（可能有多个）
        total_loss=0
        for i in range(len(ground_truth)):
            # if query_mask[i]: #如果等于1，则没有命中，该query不参与回归损失计算
            #     continue   
            true_start, true_end = ground_truth[i]
            preds = pred[i]  #pred的维度格式有问题，因为可能查询的标注区间内有多个事件，所以preds可能有多个
            # not_all_negative_indices = torch.where(torch.any(preds != -1, dim=1))[0]
            #  # 如果query_mask[i]为0，说明查询命中了，但是标注区间中不是所有的事件都命中了
            #  # 因此还是需要分别判断一下，不是所有的预测都有意义
            #  # 根据这些索引提取对应的行
            # preds = preds[not_all_negative_indices]
            
            # 计算交集的开始和结束时间
            inter_start = torch.max(preds[:, 0], true_start) #得到的是多维数据
            inter_end = torch.min(preds[:, 1], true_end)
            # 交集的长度
            inter_len = torch.clamp(inter_end - inter_start, min=0)
            # 计算并集的开始和结束时间
            union_start = torch.min(preds[:, 0], true_start)
            union_end = torch.max(preds[:, 1], true_end)
            # 并集的长度
            union_len = torch.clamp(union_end - union_start,0)
            iou = (inter_len + 1e-8) / (union_len + 1e-8)
            loss = -torch.log(iou)  #应该是个多维数据
            total_loss+=loss.mean()
        #也有可能一个都没命中，分母为0，如果全没命中的话，分子也是0，如果一个都没有命中，那么query_mask在一开始就全跳出
        # return total_loss/torch.sum(query_mask==0)+1e-8 if total_loss>0 else 0
        return total_loss/len(ground_truth)



class HighLightLoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
    
    def forward(self, labels, logits, mask, epsilon=1e-12):
        labels = labels.type(torch.float32)
        weights = torch.where(labels == 0.0, 1.0, 100.0)
        loss_per_location = nn.BCELoss(reduction='none')(logits, labels)
        loss_per_location = loss_per_location * weights
        mask = mask.type(torch.float32)
        loss = torch.sum(loss_per_location * mask) / (torch.sum(mask) + epsilon)
        return loss


class NCELoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
    
    def forward(self, labels, logits, mask, alpha):
        logits = logits / alpha
        n, d = logits.size()
        _, pos_idx = torch.max(labels, dim=1)
        pos_mask = torch.zeros_like(logits, dtype=torch.int32).to(logits.device)
        for i in range(pos_idx.size(0)):
            pos_mask[i][pos_idx[i]] = 1
        
        # neg_mask = torch.where(labels==0, 1, 0).bool()
        pos_dist = torch.masked_select(logits, mask=pos_mask.bool()).reshape(n, 1)
        neg_dist = torch.masked_select(logits, mask=(1-pos_mask).bool()).reshape(n, d-1)

        logits = torch.cat([pos_dist, neg_dist], dim=1)
        target = torch.zeros([n], dtype=torch.long, requires_grad=False).cuda()
        loss = F.cross_entropy(logits, target, reduction='mean')
        return loss