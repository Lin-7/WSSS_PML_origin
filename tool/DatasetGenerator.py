import torch
import torch.nn.functional as F
import PIL.Image
import os
import xml.etree.ElementTree as ET
from torchvision import transforms
from xml.dom.minidom import Document


def find_size (width, height): # 求生成放大图片的size（维持比例不变，宽高不小于448）
    i = 1
    w = width
    h = height

    # upsample_size=448
    upsample_size = 224
    while w < upsample_size or h < upsample_size:
    #while w < 224 or h < 224:
        i += 1
        w = width * i
        h = height * i

    return [h, w]

def get_name(box_num): # 生成新的名字
    name = str(box_num)
    leading = 5 - len(name)
    while leading >0:
        name = '0' + name
        leading -= 1
    if len(name) == 5:
        return name
    else:
        print('error')

def crop_img(img, xmin, xmax, ymin, ymax,new_img_name): # crop bounding box 并放大、保存，返回siz信息用于记录在xml里
    tran_tensor = transforms.ToTensor()
    tran_img = transforms.ToPILImage()
    box = img.crop((xmin, ymin, xmax, ymax))
    # box.show()
    width = xmax - xmin
    height = ymax - ymin
    box = tran_tensor(box)
    box = box.unsqueeze(0)
    # print(box.shape)
    size = find_size(width, height)
    box = F.interpolate(box, size=size, mode='bilinear')
    # print(box.shape)
    box = box.squeeze(0)
    box = tran_img(box)
    # box.show()
    box.save(os.path.join('Voc12_new/JPEGImages', new_img_name + '.jpg'))
    return size

def generate_xml(file_name, obj_name, size):
    #生成annotations中的xml文件。这里的xml只有代码可能用到的信息，其他VOC12中有但不一定用到的信息没有保存
    height = size[0]
    width = size[1]
    doc = Document()

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    filename = doc.createElement('filename')
    filename_text = doc.createTextNode(file_name + '.jpg')
    filename.appendChild(filename_text)
    annotation.appendChild(filename)

    sizes = doc.createElement('size')
    annotation.appendChild(sizes)

    x = doc.createElement('width')
    x_text = doc.createTextNode(str(width))
    x.appendChild(x_text)
    sizes.appendChild(x)

    x = doc.createElement('height')
    x_text = doc.createTextNode(str(height))
    x.appendChild(x_text)
    sizes.appendChild(x)

    x = doc.createElement('depth')
    x_text = doc.createTextNode('3')
    x.appendChild(x_text)
    sizes.appendChild(x)

    objects = doc.createElement('object')
    annotation.appendChild(objects)

    objname = doc.createElement('name')
    objname_text = doc.createTextNode(obj_name)
    objname.appendChild(objname_text)
    objects.appendChild(objname)

    bndbox = doc.createElement('bndbox')
    objects.appendChild(bndbox)

    x = doc.createElement('xmin')
    x_text = doc.createTextNode('0')
    x.appendChild(x_text)
    bndbox.appendChild(x)

    x = doc.createElement('xmax')
    x_text = doc.createTextNode(str(width))
    x.appendChild(x_text)
    bndbox.appendChild(x)

    x = doc.createElement('ymin')
    x_text = doc.createTextNode('0')
    x.appendChild(x_text)
    bndbox.appendChild(x)

    x = doc.createElement('ymax')
    x_text = doc.createTextNode(str(height))
    x.appendChild(x_text)
    bndbox.appendChild(x)

    with open(os.path.join('Voc12_new/Annotations', file_name + '.xml'), 'wb') as target_file:
        target_file.write(doc.toprettyxml(indent='\t', encoding='utf-8'))


if __name__ == '__main__':
    train_aug = 'train_aug.txt' #这个是原来的train_aug文件
    xml_files = 'Annotations'
    file = open(train_aug, 'r')
    box_num = 0
    os.makedirs('NewData/JPEGImages', exist_ok=True)
    os.makedirs('NewData/Annotations', exist_ok=True)

    for line in file.readlines(): #输入。从train_aug.txt读取图片名，通过图片名打开对应的图片，以及annotation信息并读取
        img_name = line[1:27]
        img = PIL.Image.open(img_name)
        # img.show()

        xml = img_name[11:-4] + '.xml'

        etree = ET.parse(os.path.join(xml_files, xml))
        root = etree.getroot()


        for obj in root.iter('object'):
            name = obj.find('name').text
            xmin = ''
            xmax = ''
            ymin = ''
            ymax = ''
            for x in obj.find('bndbox'):
                if x.tag == 'xmin':
                    xmin += x.text
                elif x.tag == 'xmax':
                    xmax += x.text
                elif x.tag == 'ymin':
                    ymin += x.text
                elif x.tag == 'ymax':
                    ymax += x.text
            xmin = int(xmin)
            xmax = int(xmax)
            ymax = int(ymax)
            try:
                ymin = int(ymin)
            except:
                ymin = ymin.split('.')[0]
                ymin = int(ymin)
            width = xmax - xmin
            height = ymax - ymin

            if width <= 110 or height <= 150: #符合条件的bounding box，处理成新图片
                box_num += 1 #只是用来生成名字的
                new_img_name = img_name[11:16] + get_name(box_num)
                size = crop_img(img, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax) #函数里crop并放大bounding box并保存图片，返回size用于生成xml
                generate_xml(new_img_name, name, size) #生成annotations中的xml文件

                # 新开一个train_aug.txt文件，记录新图片的名字
                new_train_aug = open('NewData/train_aug.txt', 'a')
                new_train_aug.write(new_img_name + '\n') # different projects may need different format of train_aug.txt


