import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tool.RoiPooling_Jointly_select import RoiPooling
import network.resnet38d

import cv2
import torchvision
import random

from scipy import stats
# from pytorch_metric_learning import miners, losses
# miner = miners.MultiSimilarityMiner()
# loss_func = losses.TripletMarginLoss()


class Net(network.resnet38d.Net):
    def __init__(self, args):
        super().__init__()

        self.args = args

        # loss
        self.loss_cls = 0
        self.loss_patch_location = 0
        self.loss_patch_cls = 0
        self.loss=0

        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.fc8_ = nn.Conv2d(256, 20, 1, bias=False)


        torch.nn.init.xavier_uniform_(self.fc8_.weight)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8_]

        # patch-based metric learning loss
        self.patch_based_metric_init()

    def patch_based_metric_init(self,):

        self.downsample = nn.Conv2d(4096, 256, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.downsample.weight)
        self.from_scratch_layers.append(self.downsample)


        self.roi_pooling = RoiPooling(mode="th", cls_layer=self.fc8_)
        self.ranking_loss = nn.MarginRankingLoss(margin=28.0)
        self.ranking_loss_same_img = nn.MarginRankingLoss(margin=28.0)

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
        iou_threshhold=0.2
        W,H = cam.size(1),cam.size(2)

        cam = F.relu(cam)
        cam=cam.mul(cls_label).cpu().numpy()

        norm_cam = cam / (np.max(cam, (1, 2), keepdims=True) + 1e-5)
        bg_score = [np.ones_like(norm_cam[0]) * bg_threshold]
        norm_cam_wbg = np.concatenate((bg_score, norm_cam))
        cam_predict = np.argmax(norm_cam_wbg, 0)

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
        return roi_index, label_list, norm_cam_wbg

    def get_roi_index_patch_same_image(self, cam, cls_label):
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
        iou_threshhold=0.2
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
                if w < 4 or h < 4 or (h / w) > 4 or (w / h) > 4:  # filter too small bounding box
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
                    if len(label_list)>3:
                        # print("2")
                        break

        # generating background object proposals
        bg_score = [np.ones_like(norm_cam[0]) * 0.05]
        cam_predict = np.argmax(np.concatenate((bg_score, norm_cam)), 0)


        # max_bg_area=2500
        # bg_box=None
        # label_i = np.zeros((W, H))
        # label_i[cam_predict == l] = 255
        # label_i = np.uint8(label_i)
        # contours, hier = cv2.findContours(label_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for c in contours:
        #     x, y, w, h = cv2.boundingRect(c)
        #     bg_area=w*h
        #     if bg_area<max_bg_area and w > 4 or h > 4:
        #         xmin = x
        #         ymin = y
        #         xmax = x + w
        #         ymax = y + h
        #
        #         bg_box=list([xmin, ymin, xmax, ymax])
        #
        # if bg_box:
        #     roi_index.append(bg_box)
        #     label_list.append(l)
        #
        # # NMS step
        # if len(label_list)==0:
        #     temp=0

        l = 0
        label_i = np.zeros((W, H))
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

                bbox_region_cam = norm_cam[l - 1, ymin:ymax, xmin:xmax]
                bbox_score = np.average(bbox_region_cam)
                if l not in bounding_box:
                    bounding_box[l] = list([[xmin, ymin, xmax, ymax]])
                    bounding_scores[l] = [bbox_score]
                else:
                    bounding_box[l].append(list([xmin, ymin, xmax, ymax]))
                    bounding_scores[l].append(bbox_score)
        # NMS step
        if l in bounding_box:
            b = torch.from_numpy(np.array(bounding_box[l], np.double)).cuda()
            s = torch.from_numpy(np.array(bounding_scores[l], np.double)).cuda()
            bounding_box_index = torchvision.ops.nms(b, s, iou_threshhold)

            for i in bounding_box_index:
                # generate new small dataset
                xmin = bounding_box[l][i][0]
                ymin = bounding_box[l][i][1]
                xmax = bounding_box[l][i][2]
                ymax = bounding_box[l][i][3]

                roi_index.append(list([xmin, ymin, xmax, ymax]))
                label_list.append(100)
                if len(label_list) > 5:
                    # print("2")
                    break

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

    def evaluate_patches(self, fg_ratio, confid=None):
        '''
        author: linqing
        date: 20230113
        params:
            fg_ratio: 前景占比的分数数组
            confid: 置信度分数数组
            args: 选择patch的相关参数

        return:
            scores: 每个patch的评估分数
        '''
        if self.args.patch_select_cri == "fgratio" or self.args.patch_select_cri == "fgAndconfid":
            if self.args.patch_select_part_fg == "front":
                fg_scores = stats.norm.pdf(fg_ratio, loc=1, scale=1)
            elif self.args.patch_select_part_fg == "mid":
                fg_scores = stats.norm.pdf(fg_ratio, loc=0.5, scale=1)
            elif self.args.patch_select_part_fg == "back":
                fg_scores = stats.norm.pdf(fg_ratio, loc=0, scale=1)
            else:
                raise ValueError(f"'patch_select_part_fg' must be front, mid or back. But now 'patch_select_part_fg' is {self.args.patch_select_part_fg}!!!!")
        
        if self.args.patch_select_cri == "confid" or self.args.patch_select_cri == "fgAndconfid":
            if self.args.patch_select_part_confid == "front":
                confid_scores = stats.norm.pdf(confid, loc=1, scale=1)
            elif self.args.patch_select_part_confid == "mid":
                confid_scores = stats.norm.pdf(confid, loc=0.5, scale=1)
            elif self.args.patch_select_part_confid == "back":
                confid_scores = stats.norm.pdf(confid, loc=0, scale=1)
            else:
                raise ValueError(f"'patch_select_part_confid' must be front, mid or back. But now 'patch_select_part_confid' is {self.args.patch_select_part_fg}!!!!")
            
        if self.args.patch_select_cri == "fgratio":
            return fg_scores
        elif self.args.patch_select_cri == "confid":
            return confid_scores
        else:
            return (0.3*fg_scores+confid_scores)/2

    def patch_based_metric_loss(self, x_patch, label, patch_num=4, is_same_img=False, is_hard_negative=True):

        N_f, _, _, _ = x_patch.size()
        cam_wo_dropout1 = self.fc8_(x_patch.detach())

        roi_cls_label = []
        roi_cls_feature_vector = []  # torch.Tensor([]).cuda()
        img_ids = []
        fg_scores = []
        proposal_num = 0 
        for i in range(N_f):# for 每张图片
            if is_same_img:
                roi_index, label_list = self.get_roi_index_patch_same_image(cam_wo_dropout1[i].detach(), label[i])
            else:
                roi_index, label_list, norm_cam_wbg = self.get_roi_index(cam_wo_dropout1[i].detach(), label[i])

            if len(label_list) > 0:
                proposal_num += len(label_list)
                roi_cls_pooled, fg_score = self.roi_pooling(x_patch[i], roi_index, label_list, patch_num, norm_cam_wbg).squeeze()  # predict roi_cls_label

                # roi_cls_pooled = torch.squeeze(roi_cls_pooled)  # [batch_num*4096]

                #########2022,增加向量归一化
                # roi_cls_pooled=roi_cls_pooled / roi_cls_pooled.norm(dim=-1, keepdim= True)

                roi_cls_feature_vector.append(roi_cls_pooled)
                roi_cls_label.extend(list(label_list) * patch_num)
                img_ids.extend([i]*len(label_list)*patch_num)
                fg_scores.extend(fg_score)

        patch_labels = torch.from_numpy(np.asarray(roi_cls_label)).cuda()
        img_ids = torch.from_numpy(np.asarray(img_ids)).cuda()
        patch_embs = torch.cat(roi_cls_feature_vector, 0)

        # ============ 挑patches ========================
        if self.args.patch_select_close:
            indexs = range(len(patch_labels))
        else:
            As = proposal_num * 4 * self.args.patch_select_ratio    # 要select的
            Ag = len(patch_labels)                                  # generate的总数
            # 以batch中的所有图片为单位挑patch
            if self.args.patch_select_cri == "random":
                indexs = np.sort(np.random.choice(range(Ag), size=int(As), replace=False, p=None))
            else:
                # 按分数(前景占比,置信度）选择指定比例的patches
                scores = self.evaluate_patches(fg_ratio=fg_scores)
                sorted_idxes = np.argsort(scores)[::-1]   # 按分数(前景占比,置信度）降序排序
                indexs = np.sort(sorted_idxes[:int(As)].copy())
        # ============ 挑patches ========================

        patch_embs_select = patch_embs[indexs]
        patch_labels_select = patch_labels[indexs]
        img_ids_select = img_ids[indexs]

        n = patch_embs_select.size(0)
        
        distance = self.euclidean_dist(patch_embs_select, patch_embs_select)

        # For each anchor, find the hardest positive and negative
        mask = patch_labels_select.expand(n, n).eq(patch_labels_select.expand(n, n).t())
        img_mask = img_ids_select.expand(n, n).eq(img_ids_select.expand(n, n).t())

        dist_ap, dist_an = [], []
        
        for i in range(n):
            if patch_labels_select[i].item()==100:
                continue
            if is_same_img:
                # neg_idx=~mask[i] & img_mask[i]
                neg_idx=~mask[i]
            else:
                neg_idx = ~mask[i]
            an_i = distance[i][neg_idx]
            if an_i.size(0) == 0:
                continue
            if is_same_img:
                # pos_idx=mask[i] & img_mask[i] # 这个限制了学习图片之前的同类别相似性
                pos_idx=mask[i]
            else:
                pos_idx=mask[i]
            if is_hard_negative:
                dist_ap.append(distance[i][pos_idx].max().unsqueeze(0))
                dist_an.append(an_i.min().unsqueeze(0))
            else:
                # dist_ap.append(distance[i][pos_idx].topk(4).values[random.randint(0, 3)].unsqueeze(0))
                # dist_an.append(an_i.topk(4, largest=False).values[random.randint(0, 3)].unsqueeze(0))
                dist_ap.append(distance[i][pos_idx][random.randint(0, distance[i][pos_idx].shape[0]-1)].unsqueeze(0))
                dist_an.append(an_i[random.randint(0, an_i.shape[0]-1)].unsqueeze(0))

        # triplet loss
        if len(dist_ap)>0:
            dist_ap = torch.cat(dist_ap, 0)
            dist_an = torch.cat(dist_an, 0)

            y = torch.ones_like(dist_an)

            if is_same_img:
                loss_patch_cls = self.ranking_loss_same_img(dist_an, dist_ap, y) / y.shape[0]
            else:
                loss_patch_cls = self.ranking_loss(dist_an, dist_ap, y) / y.shape[0]

        # # ================
            return loss_patch_cls
        else:
            return torch.tensor(0.0).cuda()

    def forward(self, x, bounding_box=None, label=None, param=None, is_patch_metric=True, patch_num=4,is_sse=False):
        N, C, H, W = x.size()
        # keep the feature map without dropout
        x = super().forward(x)
        x = self.dropout7(x)
        x = self.downsample(x)
        x_patch = F.relu(x)
        x2 = self.fc8_(x_patch)

        cam = F.interpolate(x2, (H, W), mode='bilinear')

        if label is not None:
            # multi-label soft margin loss:
            predicts = F.adaptive_avg_pool2d(cam, (1, 1))  # GAP的作用，得到各类置信度
            loss_cls = F.multilabel_soft_margin_loss(predicts, label)


            result = [self.loss_cls]

            if is_patch_metric:
                patch_metric_loss=self.patch_based_metric_loss(x_patch, label, is_same_img=False, is_hard_negative=True)
                # patch_metric_loss=self.patch_based_metric_loss_cam(x2, label, is_same_img=True)
                # patch_metric_loss_9=self.patch_based_metric_loss(x_wo_dropout, label,patch_num=9,)
                # patch_metric_loss_same_img=self.patch_based_metric_loss(x_wo_dropout, label, is_same_img=True)
                # patch_loss= (patch_metric_loss+patch_metric_loss_same_img)/2
                # patch_metric_loss= patch_metric_loss_same_img
                # patch_metric_loss= patch_metric_loss_same_img
                return [loss_cls, patch_metric_loss]

        else:
            result = cam

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
