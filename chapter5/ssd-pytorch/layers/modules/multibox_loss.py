# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import voc as cfg
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap

        #import pdb
        #pdb.set_trace()

        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        # 1 首先匹配正负样本
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            # 得到每一个prior对应的truth,放到loc_t与conf_t中,conf_t中是类别,loc_t中是[matches, prior, variance]
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets

        # 2 计算所有正样本的定位损失,负样本不需要定位损失

        # 计算正样本的数量
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        #import pdb
        #pdb.set_trace()


        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        # 将pos_idx扩展为[32, 8732, 4],正样本的索引
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # 正样本的定位预测值
        loc_p = loc_data[pos_idx].view(-1, 4)
        # 正样本的定位真值
        loc_t = loc_t[pos_idx].view(-1, 4)
        # 所有正样本的定位损失
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # 3 对于类别损失,进行难样本挖掘,控制比例为1:3

        # Compute max conf across batch for hard negative mining
        # 所有prior的类别预测
        batch_conf = conf_data.view(-1, self.num_classes)
        # 计算类别损失.每一个的log(sum(exp(21个的预测)))-对应的真正预测值
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(pos.size()[0], pos.size()[1])
        # 首先过滤掉正样本
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        # idx_rank为排序后每个元素的排名
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)

        # 这个地方负样本的最大值不应该是pos.size(1)-num_pos?
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        # 选择每个batch中负样本的索引
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # 4 计算正负样本的类别损失

        # Confidence Loss Including Positive and Negative Examples
        # 都扩展为[32, 8732, 21]
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # 把预测提出来
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        # 对应的标签
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N.type('torch.cuda.FloatTensor')
        loss_c /= N.type('torch.cuda.FloatTensor')
        return loss_l, loss_c
