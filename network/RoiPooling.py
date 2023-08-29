from torch.nn.modules.module import Module
import numpy as np
import torch

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
            pool = np.zeros((region_channels, pool_height, pool_width))
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

                if xmin==xmax or ymin==ymax:
                    continue
                if self.mode=='tf':
                    pool[i, j, :] = np.max(region[ymin:ymax, xmin:xmax, :], axis=(0,1))
                elif self.mode=='th':

                    region_var = region[:, ymin:ymax, xmin:xmax]
                    pool[:, i, j] = np.max(region_var.cpu().detach().numpy(), axis=(1,2))
                    # pool[:, i, j] = torch.max(region[:, ymin:ymax, xmin:xmax], axis=(1, 2))

        # pool = torch.from_numpy(pool)
        return pool

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
            r = np.squeeze(feature_map)[:, ymin:ymax, xmin:xmax]
        return r

    def get_pooled_rois(self,feature_map, roi_batch):
        """
        getting pools from the roi batch
        :param feature_map:
        :param roi_batch: region of interest batch (usually is 256 for faster rcnn)
        :return:
        """
        pool = []
        i = 0
        for region_dim in roi_batch:
            map = feature_map[i]
            # print(map.shape)
            # print(region_dim)
            region = self.get_region(map, region_dim)
            p = self.pool(region)
            pool.append(p)
            i += 1
        pool = np.array(pool)
        pool = torch.from_numpy(pool)
        pool = pool.cuda()
        # print(pool.shape)
        return pool

    def forward(self,feature_map, roi_batch, inner_batch_size):
        """
        getting pools from the roi batch
        :param feature_map:
        :param roi_batch: region of interest batch (usually is 256 for faster rcnn)
        :return:
        """
        pool = []
        i = 0
        roi_num = 0
        for region_dim in roi_batch:
            map = feature_map[i]
            # print("map: ", map.shape)
            # print(region_dim)
            region = self.get_region(map, region_dim)
            p = self.pool_region(region)
            pool.append(p)
            roi_num +=1
            if roi_num >sum(inner_batch_size[:i+1]):
                i+=1
        pool = np.array(pool)
        pool = torch.from_numpy(pool)
        pool = pool.cuda()
        # print(pool.shape)
        return pool