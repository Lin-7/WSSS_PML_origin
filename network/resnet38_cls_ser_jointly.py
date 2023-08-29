import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tool.RoiPooling_Jointly import RoiPooling
import network.resnet38d

import cv2
import torchvision
import random

from pytorch_metric_learning import miners, losses
miner = miners.MultiSimilarityMiner()
loss_func = losses.TripletMarginLoss()


class Net(network.resnet38d.Net):
    def __init__(self):
        super().__init__()

        # loss
        self.loss_cls = 0
        self.loss_patch_location = 0
        self.loss_patch_cls = 0
        self.loss=0

        self.dropout7 = torch.nn.Dropout2d(0.5)
        self.fc8 = nn.Conv2d(4096, 20, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8.weight)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8]

        self.roi_pooling = RoiPooling(mode="th")

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.ranking_loss = nn.MarginRankingLoss(margin=28.0)
        ########2022
        # self.ranking_loss = nn.MarginRankingLoss(margin=0.5)


    def get_roi_index(self, cam, cls_label):
        '''
        For each image
        :param cam: 20 * W* H
        :param cls_label:
        :return:
        roi_index[N,4]
        4->[XMIN,YMIN,XMAX,YMAX]

        label=[N]

        limitation:
        '''
        bg_threshold=0.20
        iou_threshhold=0.3
        W,H = cam.size(1),cam.size(2)

        cam = F.relu(cam)
        cam=cam.mul(cls_label).cpu().numpy()

        norm_cam = cam / (np.max(cam, (1, 2), keepdims=True) + 1e-5)
        bg_score = [np.ones_like(norm_cam[0]) * bg_threshold]
        cam_predict = np.argmax(np.concatenate((bg_score, norm_cam)), 0)

        label = np.unique(cam_predict)
        label = label[1:]  # get the label except background
        # for each class

        bounding_box = {}
        bounding_scores = {}

        roi_index=[]
        label_list=[]

        for l in label:
            label_i = np.zeros((W,H))
            label_i[cam_predict == l] = 255
            label_i = np.uint8(label_i)
            contours, hier = cv2.findContours(label_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w < 9 or h < 9 or (h / w) > 4 or (w / h) > 4:  # filter too small bounding box
                    continue
                else:
                    xmin = x
                    ymin = y
                    xmax = x + w
                    ymax = y + h

                    bbox_region_cam = norm_cam[l-1, ymin:ymax, xmin:xmax]
                    bbox_score = np.average(bbox_region_cam)
                    if l not in bounding_box:
                        bounding_box[l] = list([[xmin, ymin, xmax, ymax]])
                        bounding_scores[l] = [bbox_score]
                    else:
                        bounding_box[l].append(list([xmin, ymin, xmax, ymax]))
                        bounding_scores[l].append(bbox_score)
            # NMS step
            if l in bounding_box:
                b = torch.from_numpy(np.array(bounding_box[l],np.double)).cuda()
                s = torch.from_numpy(np.array(bounding_scores[l],np.double)).cuda()
                bounding_box_index = torchvision.ops.nms(b,s,iou_threshhold)

                for i in bounding_box_index:
                    # generate new small dataset
                    xmin = bounding_box[l][i][0]
                    ymin = bounding_box[l][i][1]
                    xmax = bounding_box[l][i][2]
                    ymax = bounding_box[l][i][3]

                    roi_index.append(list([xmin, ymin, xmax, ymax]))
                    label_list.append(l)
                    if len(label_list)>=3:
                        # print("2")
                        return roi_index, label_list
        return roi_index, label_list

    def euclidean_dist(self,x, y):
        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """

        m, n = x.size(0), y.size(0)
        # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        # yy会在最后进行转置的操作
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
        dist.addmm_(1, -2, x, y.t())
        # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist



    def forward(self, x, bounding_box=None, label=None, param=None, is_patch_metric=True, patch_num=4,is_sse=False):
        N, C, H, W = x.size()
        x = super().forward(x)
        x = self.dropout7(x)

        x2 = self.fc8(x)
        cam = F.interpolate(x2,(H,W),mode='bilinear')

        N_f, C_f, H_f, W_f = cam.size()

        if label is not None:
            # multi-label soft margin loss:
            predicts = F.adaptive_avg_pool2d(cam, (1, 1))  # GAP的作用，得到各类置信度
            self.loss_cls = F.multilabel_soft_margin_loss(predicts, label)

            if is_patch_metric:
                roi_cls_label=[]
                roi_cls_feature_vector=[] #torch.Tensor([]).cuda()
                for i in range(N_f):
                    roi_index, label_list=self.get_roi_index(x2[i].detach(), label[i])
                    if len(label_list)>0:
                        roi_cls_pooled = self.roi_pooling(x[i], roi_index, patch_num).cuda()  # predict roi_cls_label

                        roi_cls_pooled = torch.squeeze(roi_cls_pooled)  # [batch_num*4096]

                        #########2022,增加向量归一化
                        # roi_cls_pooled=roi_cls_pooled / roi_cls_pooled.norm(dim=-1, keepdim= True)

                        roi_cls_feature_vector.append(roi_cls_pooled)

                        roi_cls_label.extend(list(label_list)*patch_num)

                patch_labels = torch.from_numpy(np.asarray(roi_cls_label)).cuda()

                patch_embs = torch.cat(roi_cls_feature_vector,0)

                n = patch_embs.size(0)
                # =============== euclidean distance + triplet loss with hard sample mining ===============

                # try to use metric learning lib directly : the function below use cosine similarity
                # hard_pairs = miner(patch_embs, patch_labels)
                # self.loss_patch_cls = loss_func(patch_embs, patch_labels, hard_pairs)


                # ======
                distance=self.euclidean_dist(patch_embs, patch_embs)

                # For each anchor, find the hardest positive and negative
                mask = patch_labels.expand(n, n).eq(patch_labels.expand(n, n).t())
                dist_ap, dist_an = [], []

                is_hard_negative = True


                for i in range(n):
                    an_i = distance[i][mask[i] == 0]
                    if an_i.size(0) == 0:
                        continue
                    if is_hard_negative:
                        dist_ap.append(distance[i][mask[i]].max().unsqueeze(0))
                        dist_an.append(an_i.min().unsqueeze(0))
                    else:
                        dist_ap.append(distance[i][mask[i]].topk(4).values[random.randint(0,3)].unsqueeze(0))
                        dist_an.append(an_i.topk(4, largest=False).values[random.randint(0,3)].unsqueeze(0))


                # triplet loss
                dist_ap = torch.cat(dist_ap,0)
                dist_an = torch.cat(dist_an,0)
                # print(dist_ap[:10])
                # print(dist_an[:10])
                y = torch.ones_like(dist_an)

                self.loss_patch_cls = self.ranking_loss(dist_an, dist_ap, y)/y.shape[0]
                # 2022 这个loss不需要除以batch size吧
                # self.loss_patch_cls = self.ranking_loss(dist_an, dist_ap, y)

                # # ================

                return [self.loss_cls,self.loss_patch_cls]

            result = [self.loss_cls]

        else:

            result = [cam]

        return result


    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups
