#!/usr/bin/python
# -*- coding: UTF-8 -*-
# get annotation object bndbox location
import os
import cv2
import PIL.Image
import numpy as np
try:
    import xml.etree.cElementTree as ET  #解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']
CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))
										         
##get object annotation bndbox loc start 
def GetAnnotBoxLoc(AnotPath):#AnotPath VOC标注文件路径
    tree = ET.ElementTree(file=AnotPath)  #打开文件，解析成一棵树型结构
    root = tree.getroot()#获取树型结构的根
    ObjectSet=root.findall('object')#找到文件中所有含有object关键字的地方，这些地方含有标注目标
    ObjBndBoxSet={} #以目标类别为关键字，目标框为值组成的字典结构
    for Object in ObjectSet:
        ObjName=Object.find('name').text
        BndBox=Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text)#-1 #-1是因为程序是按0作为起始位置的
        y1 = int(BndBox.find('ymin').text)#-1
        x2 = int(BndBox.find('xmax').text)#-1
        y2 = int(BndBox.find('ymax').text)#-1
        BndBoxLoc=[x1,y1,x2,y2]
        cat_num = CAT_NAME_TO_NUM[ObjName]+1
        if cat_num in ObjBndBoxSet:
        	ObjBndBoxSet[cat_num].append(BndBoxLoc)#如果字典结构中含有这个类别了，那么这个目标框要追加到其值的末尾
        else:
        	ObjBndBoxSet[cat_num]=[BndBoxLoc]#如果字典结构中没有这个类别，那么这个目标框就直接赋值给其值吧
    return ObjBndBoxSet
##get object annotation bndbox loc end

def display(objBox,pic):
    img = cv2.imread(pic)
    
    for key in objBox.keys():
        for i in range(len(objBox[key])):
            cv2.rectangle(img, (objBox[key][i][0],objBox[key][i][1]), (objBox[key][i][2], objBox[key][i][3]), (0, 0, 255), 2)        
            cv2.putText(img, key, (objBox[key][i][0],objBox[key][i][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
    cv2.imshow('img',img)
    cv2.imwrite('display.jpg',img)
    cv2.waitKey(0)


if __name__== '__main__':
  #   pic = r"F:/学习/Patrick/image segmentation/参考工程/DSRG-tensorflow-master/data/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg"
  #   train_file_path = "F:/学习/Patrick/image segmentation/参考工程/SSENet-pytorch-master/voc12/train_aug.txt"
  #   train_aug_file = open(train_file_path, "r")
  #   bounding_box_list = {}
  #   for line in train_aug_file.readlines():
  #   	line = line[12:23]
		# # train_aug_list.append(line)
  #   	ObjBndBoxSet=GetAnnotBoxLoc(os.path.join('F:/学习/Patrick/image segmentation/参考工程/DSRG-tensorflow-master/data/VOCdevkit/VOC2012/Annotations','%s.xml'%line))
  #   	bounding_box_list[line] = ObjBndBoxSet

  #   val_file_path = "F:/学习/Patrick/image segmentation/参考工程/SSENet-pytorch-master/voc12/val.txt"
  #   val_file = open(val_file_path, "r")
  #   for line in val_file.readlines():
  #   	line = line[12:23]
		# # train_aug_list.append(line)
  #   	ObjBndBoxSet=GetAnnotBoxLoc(os.path.join('F:/学习/Patrick/image segmentation/参考工程/DSRG-tensorflow-master/data/VOCdevkit/VOC2012/Annotations','%s.xml'%line))
  #   	bounding_box_list[line] = ObjBndBoxSet

  #   print(ObjBndBoxSet)
  #   np.save("bounding_box.npy", bounding_box_list)
    


    bounding = np.load(os.path.join('bounding_box.npy'), allow_pickle=True).item()
    bounding_box = bounding['2007_000033']
    print(bounding_box)
  # # #   for key in bounding_box.keys():
  # #   	print(key)
		# # lens = len(bounding_box[key])
		# # for i in range(lens):
		# # 	print(bounding_box[key][i])
