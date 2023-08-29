from torch.nn.modules.module import Module
import numpy as np
import torch
import torch.nn.functional as F
import time
import math

class RoiPooling(Module):
    def __init__(self, mode='tf', pool_size=(1, 1)):
        """
        tf: (height, width, channels)
        th: (channels, height, width)
        :param mode:
        :param pool_size:
        """
        super(RoiPooling, self).__init__()

        self.mode = mode
        self.pool_size = pool_size

    def pool_region_ori(self, region):

        """
        the pooling of a region
        :param region: the region of interest fetched from feature map
        :return: roipool with size of (1, height, width, channel) if mode is tf otherwise (1, channels, height, width)
        """

        pool_height, pool_width = self.pool_size
        if self.mode == 'tf':
            region_height, region_width, region_channels = region.shape
            pool = np.zeros((pool_height, pool_width, region_channels))
        elif self.mode== 'th':
            region_channels, region_height, region_width = region.shape
            pool = torch.from_numpy(np.zeros((region_channels, pool_height, pool_width)))

        h_step = region_height / pool_height
        w_step = region_width / pool_width
        for i in range(pool_height):
            for j in range(pool_width):

                xmin = j * w_step
                xmax = (j+1) * w_step
                ymin = i * h_step
                ymax = (i+1) * h_step

                xmin = int(xmin)
                xmax = int(xmax)
                ymin = int(ymin)
                ymax = int(ymax)

                # if xmin==xmax or ymin==ymax:
                #     continue
                if self.mode=='tf':
                    # pool[i, j, :] = np.max(region[ymin:ymax, xmin:xmax, :], axis=(0,1))
                    pool[i,j,:] = F.adaptive_avg_pool2d(region[ymin:ymax, xmin:xmax, :], (1,1))
                    # pool[i, j, :] = F.adaptive_max_pool2d(region[ymin:ymax, xmin:xmax, :], (1, 1))
                elif self.mode=='th':

                    region_var = region[:, ymin:ymax, xmin:xmax]
                    # pool[:, i, j] = np.max(region_var.cpu().detach().numpy(), axis=(1,2))
                    roi_pooled = F.adaptive_avg_pool2d(region_var, (1, 1))

                    # # roi_pooled = F.adaptive_max_pool2d(region_var, (1, 1))
                    # # roi_pooled_a = roi_pooled[:,0,0]
                    # # pool_a  = pool[:, i,j]
                    #
                    #
                    # pool[:, i, j] = roi_pooled[:, 0, 0]


        # pool = torch.from_numpy(pool)
        return roi_pooled

    def pool_region(self, region):

        """
        the pooling of a region
        :param region: the region of interest fetched from feature map
        :return: roipool with size of (1, height, width, channel) if mode is tf otherwise (1, channels, height, width)
        """

        pool_height, pool_width = self.pool_size
        if self.mode == 'tf':
            region_height, region_width, region_channels = region.shape
            pool = np.zeros((pool_height, pool_width, region_channels))
        elif self.mode== 'th':
            region_channels, region_height, region_width = region.shape
            pool = torch.from_numpy(np.zeros((region_channels, pool_height, pool_width)))

        # h_step = region_height / pool_height
        # w_step = region_width / pool_width
        # for i in range(pool_height):
        #     for j in range(pool_width):
        #
        #         xmin = j * w_step
        #         xmax = (j+1) * w_step
        #         ymin = i * h_step
        #         ymax = (i+1) * h_step
        #
        #         xmin = int(xmin)
        #         xmax = int(xmax)
        #         ymin = int(ymin)
        #         ymax = int(ymax)
        #
        #         if xmin==xmax or ymin==ymax:
        #             continue
        if self.mode=='th':
            # pool[:, i, j] = np.max(region_var.cpu().detach().numpy(), axis=(1,2))
            roi_pooled = F.adaptive_avg_pool2d(region, (1 ,1))

                    # # roi_pooled = F.adaptive_max_pool2d(region_var, (1, 1))
                    # # roi_pooled_a = roi_pooled[:,0,0]
                    # # pool_a  = pool[:, i,j]
                    #
                    #
                    # pool[:, i, j] = roi_pooled[:, 0, 0]


        # pool = torch.from_numpy(pool)
        return roi_pooled


    def get_region(self, feature_map, roi_dimensions):
        """
        fetching the roi from feature map by the dimension of the roi
        :param feature_map: the feature map with size of (1, height, width, channels)
        :param roi_dimensions: a region of interest dimensions
        :return:
        """
        xmin, ymin, xmax, ymax = roi_dimensions
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        if self.mode=='tf':
            r = np.squeeze(feature_map)[ymin:ymax, xmin:xmax, :]
        elif self.mode=='th':
            r = feature_map[:, ymin:ymax, xmin:xmax]
        return r

    # def get_pooled_rois(self,feature_map, roi_batch):
    #     """
    #     getting pools from the roi batch
    #     :param feature_map:
    #     :param roi_batch: region of interest batch (usually is 256 for faster rcnn)
    #     :return:
    #     """
    #     pool = []
    #     i = 0
    #     for region_dim in roi_batch:
    #         map = feature_map[i]
    #         # print(map.shape)
    #         # print(region_dim)
    #         region = self.get_region(map, region_dim)
    #         p = self.pool(region)
    #         pool.append(p)
    #         i += 1
    #     pool = np.array(pool)
    #     pool = torch.from_numpy(pool)
    #     pool = pool.cuda()
    #     # print(pool.shape)
    #     return pool

    def forward(self,feature_map, roi_batch, patch_num):
        """
        getting pools from the roi batch
        :param feature_map:
        :param roi_batch: region of interest batch (usually is 256 for faster rcnn)
        :return:
        """
        # start=time.time()
        pool = []
        # pool = torch.Tensor()
        i = 0
        roi_num = 0

        patch_num_per_edge = int(math.sqrt(patch_num))

        pool=[]
        for i in range(patch_num):

            for region_dim in roi_batch:
                map = feature_map
                xmin,ymin,xmax,ymax=region_dim

                region_dim_i=[round(xmin+ (i % patch_num_per_edge) * (xmax - xmin) / patch_num_per_edge),
                              round(ymin + int(i / patch_num_per_edge) * (ymax - ymin) / patch_num_per_edge),
                              round(xmin+ (i % patch_num_per_edge) * (xmax - xmin) / patch_num_per_edge+(xmax - xmin) / patch_num_per_edge),
                              round(ymin + int(i / patch_num_per_edge) * (ymax - ymin) / patch_num_per_edge+(ymax - ymin) / patch_num_per_edge)
                              ]
                # print(region_dim_i)

                region = self.get_region(map, region_dim_i)
                p = self.pool_region_ori(region).unsqueeze(0)

                pool.append(p)

        pool=torch.cat(pool,dim=0)

        # pool = pool.cuda()

        return pool