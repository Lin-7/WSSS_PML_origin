import torch
import numpy as np
import random
torch.manual_seed(1234) # cpu
torch.cuda.manual_seed(1234) #gpu
np.random.seed(1234) #numpy
random.seed(1234) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils, torchutils, visualization
import argparse
import importlib
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
import shutil

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']


def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=24, type=int)
    parser.add_argument("--max_epoches", default=3, type=int)
    parser.add_argument("--network", default="network.resnet38_cls_ser_jointly_revised_seperatable", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)

    parser.add_argument("--weights", default="/usr/volume/WSSS/weights_released/res38_cls.pth", type=str)
    # parser.add_argument("--weights", default="/usr/volume/WSSS/weights_released/resnet38_cls_ser_0.3.pth", type=str)

    parser.add_argument("--voc12_root", default="/usr/volume/WSSS/VOCdevkit/VOC2012", type=str)

    parser.add_argument("--train_list", "-tr", default="/usr/volume/WSSS/WSSS_PML_origin/voc12/train_aug.txt", type=str)
    # parser.add_argument("--train_voc_list", "-trvoc", default="/usr/volume/WSSS/WSSS_PML_origin/voc12/train_voc12.txt", type=str)
    # parser.add_argument("--val_list", default="/usr/volume/WSSS/WSSS_PML_origin/voc12/val.txt", type=str)
    parser.add_argument("--tensorboard_img", default="/usr/volume/WSSS/WSSS_PML_origin/voc12/tensorborad_img.txt", type=str)

    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--optimizer", default='poly', type=str)

    parser.add_argument("--session_name", default="base", type=str)
    parser.add_argument("--tblog_dir", default="./saved_checkpoints", type=str)
    # 模型保存地址：# tblog_dir/session_name/


    args = parser.parse_args()


    log_root = f"/usr/volume/WSSS/WSSS_PML_origin/{args.tblog_dir}/{args.session_name}/"
    os.makedirs(log_root, exist_ok=True)
    if os.path.exists(log_root):
        shutil.rmtree(log_root)

    #### train from imagenet params
    # args.session_name="from_imageNet"
    # args.weights="/usr/volume/WSSS/WSSS_PML_origin/weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params"
    # args.lr=0.1
    #
    # args.optimizer="poly"
    #### train from imagenet params

    model = getattr(importlib.import_module(args.network), 'Net')()

    tblogger = SummaryWriter(args.tblog_dir+'log')
    pyutils.Logger(args.session_name + '.log')

    print(vars(args))
    w, h = [448, 448]

    # dataset
    train_dataset = voc12.data.VOC12ClsDataset(args.train_list, voc12_root=args.voc12_root,
                                               transform=transforms.Compose([
                                                   imutils.RandomResizeLong(448, 768),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                                                                          hue=0.1),
                                                   np.asarray,
                                                   model.normalize,
                                                   imutils.RandomCrop(args.crop_size),
                                                   imutils.HWC_to_CHW,
                                                   torch.from_numpy
                                               ]))

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                   worker_init_fn=worker_init_fn)


    max_step = (len(train_dataset) // args.batch_size)*args.max_epoches

    tensorboard_dataset = voc12.data.VOC12ClsDataset(args.tensorboard_img, voc12_root=args.voc12_root,
                                                     transform=transforms.Compose([
                                                         np.asarray,
                                                         model.normalize,
                                                         imutils.CenterCrop(500),
                                                         imutils.HWC_to_CHW,
                                                         torch.from_numpy
                                                     ]))
    tensorboard_img_loader = DataLoader(tensorboard_dataset,
                                        shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)


    param_groups = model.get_parameter_groups()
    if args.optimizer=='poly':
        optimizer = torchutils.PolyOptimizer([
            {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
            {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
            {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
            {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
        ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)
    elif args.optimizer=='adam':
        optimizer=torchutils.Adam([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights[-7:] == '.params':
        assert args.network == "network.resnet38_cls_ser_jointly"
        import network.resnet38d
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    model=torch.nn.DataParallel(model).cuda()


    model.train()

    avg_meter = pyutils.AverageMeter('loss')
    avg_meter1 = pyutils.AverageMeter('loss_cls')
    avg_meter2 = pyutils.AverageMeter('loss_patch')

    timer = pyutils.Timer("Session started: ")

    loss_list=[]
    validation_set_CAM_mIoU=[]
    val_loss_list = []
    train_set_CAM_mIoU = []
    val_multi_mIoU = []
    train_multi_mIoU = []

    global_step = 0
    patch_num = 0
    max_step_small = 0

    is_opti = True

    is_need_load_proposal_data=False
    is_init_opti = False
    train_small_dataset=None
    train_small_data_loader=None
    optimizer_patch_cls = None


    # training
    for ep in range(args.max_epoches):
        itr = ep + 1
        # log the images at the beginning of each epoch
        # for iter, pack in enumerate(tensorboard_img_loader):
        #     tensorboard_img = pack[1]
        #     tensorboard_label = pack[2].cuda(non_blocking=True)
        #     tensorboard_label = tensorboard_label.unsqueeze(2).unsqueeze(3)
        #     N, C, H, W = tensorboard_img.size()
        #     img_8 = tensorboard_img[0].numpy().transpose((1, 2, 0))
        #     img_8 = np.ascontiguousarray(img_8)
        #     mean = (0.485, 0.456, 0.406)
        #     std = (0.229, 0.224, 0.225)
        #     img_8[:, :, 0] = (img_8[:, :, 0] * std[0] + mean[0]) * 255
        #     img_8[:, :, 1] = (img_8[:, :, 1] * std[1] + mean[1]) * 255
        #     img_8[:, :, 2] = (img_8[:, :, 2] * std[2] + mean[2]) * 255
        #     img_8[img_8 > 255] = 255
        #     img_8[img_8 < 0] = 0
        #     img_8 = img_8.astype(np.uint8)

        #     input_img = img_8.transpose((2, 0, 1))
        #     h = H // 4
        #     w = W // 4
        #     model.eval()
        #     with torch.no_grad():
        #         cam = model(tensorboard_img)
        #     model.train()
        #     p = F.interpolate(cam, (h, w), mode='bilinear')[0].detach().cpu().numpy()
        #     bg_score = np.zeros((1, h, w), np.float32)
        #     p = np.concatenate((bg_score, p), axis=0)
        #     bg_label = np.ones((1, 1, 1), np.float32)
        #     l = tensorboard_label[0].detach().cpu().numpy()
        #     l = np.concatenate((bg_label, l), axis=0)
        #     image = cv2.resize(img_8, (w, h), interpolation=cv2.INTER_CUBIC).transpose((2, 0, 1))
        #     CLS, CAM, CLS_crf, CAM_crf = visualization.generate_vis(p, l, image,
        #                                                             func_label2color=visualization.VOClabel2colormap)
        #     tblogger.add_image('Image_' + str(iter), input_img, itr)
        #     tblogger.add_image('CLS_' + str(iter), CLS, itr)
        #     tblogger.add_image('CLS_crf' + str(iter), CLS_crf, itr)
        #     tblogger.add_images('CAM_' + str(iter), CAM, itr)


        #     # print("Epoch %s: " % str(ep), "%.2fs" % (timer.get_stage_elapsed()))

        #     timer.reset_stage()

        for iter, pack in tqdm(enumerate(train_data_loader)):

            name = pack[0]
            img = pack[1]
            label = pack[2].cuda(non_blocking=True)
            label = label.unsqueeze(2).unsqueeze(3)
            raw_H = pack[3]
            raw_W = pack[4]
            pack3 = []
            param = [w, h, raw_W, raw_H, patch_num]

            optimizer.zero_grad()
            loss_cls, loss_patch = model(x=img, label=label, param=param, is_patch_metric=True, is_sse=False)

            loss_cls=loss_cls.mean()

            loss_patch=loss_patch.mean()
                # loss=loss_cls+loss_patch/20
                ### 2022
            loss = loss_cls + loss_patch / 10
            avg_meter2.add({'loss_patch': loss_patch.item()})


            avg_meter.add({'loss': loss.item()})
            avg_meter1.add({'loss_cls': loss_cls.item()})


            loss.backward()
            optimizer.step()

            global_step+=1

            # print
            if (global_step-1)%10 == 0:
                timer.update_progress(global_step / max_step)

                a=avg_meter.get('loss')
                loss_list.append(avg_meter.get('loss'))

                print('Iter:%5d/%5d' % (global_step - 1, max_step),
                      'Loss_cls: %.4f' % (avg_meter.get('loss')),
                      'Loss_cls: %.4f:'%(avg_meter1.get('loss_cls')),
                      'Loss_patch: %.4f:' % (avg_meter2.get('loss_patch')),
                      'imps:%.3f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.6f' % (optimizer.param_groups[0]['lr']), flush=True)
                avg_meter.pop()

        print(f"epoch{ep} end!!!!!!!!!!!!!!!!!!!")
        if args.optimizer=='adam':
            optimizer.adam_turn_step()

        model_saved_root=f"/usr/volume/WSSS/WSSS_PML_origin/{args.tblog_dir}/{args.session_name}/"
        os.makedirs(model_saved_root, exist_ok=True)
        model_saved_dir = os.path.join(model_saved_root, f"{ep}ep.pth")

        torch.save(model.module.state_dict(),model_saved_dir)
        avg_meter.pop()

        # evaluation

        loss_dict = {'loss': loss_list[-1]}
        tblogger.add_scalars('cls_loss', loss_dict, itr)
        tblogger.add_scalar('cls_lr', optimizer.param_groups[0]['lr'], itr)

        # tensorboard log vis images

        result_saved_dir=os.path.join(f"{log_root}/log_txt/", f"{args.session_name}_{ep}")
        os.makedirs(f"{log_root}/log_txt/", exist_ok=True)
        # import pdb; pdb.set_trace()
        os.system(f"/home/vipuser/anaconda3/envs/mywsss/bin/python infer_cls_pml.py --log_infer_cls={result_saved_dir}.txt --weights={model_saved_dir} --out_cam_pred=./out_cam_ser_{args.session_name}")


    # np.save('loss.npy', loss_list)
    # np.save('validation_set_CAM_mIoU.npy', validation_set_CAM_mIoU)
