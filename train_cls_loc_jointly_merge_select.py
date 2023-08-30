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
import torchvision
import voc12.data
from tool import pyutils, imutils, torchutils, visualization
import argparse
import importlib
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
import shutil
from PIL import Image
import time

from evaluation import eval

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']


def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_epoches", default=4, type=int)
    parser.add_argument("--network", default="network.resnet38_cls_select", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--patch_loss_weight", default=0.2, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--infer_num_workers", default=12, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--optimizer", default='poly', type=str)

    parser.add_argument("--weights", default="/usr/volume/WSSS/weights_released/res38_cls.pth", type=str)
    parser.add_argument("--voc12_root", default="/usr/volume/WSSS/VOCdevkit/VOC2012", type=str)

    parser.add_argument("--train_list", "-tr", default="/usr/volume/WSSS/WSSS_PML_origin/voc12/train_aug.txt", type=str)
    # parser.add_argument("--train_voc_list", "-trvoc", default="/usr/volume/WSSS/WSSS_PML_origin/voc12/train_voc12.txt", type=str)
    # parser.add_argument("--val_list", default="/usr/volume/WSSS/WSSS_PML_origin/voc12/val.txt", type=str)
    parser.add_argument("--tensorboard_img", default="/usr/volume/WSSS/WSSS_PML_origin/voc12/tensorborad_img.txt", type=str)
    parser.add_argument("--infer_list", default=f"/usr/volume/WSSS/WSSS_PML_origin/voc12/val_voc12.txt", type=str)

    parser.add_argument("--patch_select_close", default=False, type=bool)   # 不选择patches
    parser.add_argument("--patch_select_cri", default="random", type=str)   # fgratio confid fgAndconfid random
    parser.add_argument("--patch_select_ratio", default=0.3, type=float)   # 0.3 0.4 0.5
    parser.add_argument("--patch_select_part_fg", default="mid", type=str)   # front mid back
    parser.add_argument("--patch_select_part_confid", default="front", type=str)   # front mid back

    parser.add_argument("--out_cam", default="./out_cam", type=str)
    parser.add_argument("--out_crf", default="./out_crf", type=str)

    parser.add_argument("--session_name", default="base-bs32", type=str)
    parser.add_argument("--tblog_dir", default="./saved_checkpoints", type=str)
    # 模型保存地址：# tblog_dir/session_name/

    args = parser.parse_args()
    
    phase = "val"
    bg_thresh=[0.21,0.22,0.23,0.24,0.243,0.245,0.248,0.25]
    crf_alpha = [4, 16, 24, 28, 32]

    args.out_cam_pred=f"./out_cam_ser_{args.session_name}"
    
    # saved_checkpoints下有每个session的文件，记录每个epoch的训练效果和对应的模型
    log_root = f"/usr/volume/WSSS/WSSS_PML_origin/{args.tblog_dir}/{args.session_name}/"
    if os.path.exists(log_root):
        shutil.rmtree(log_root)
    os.makedirs(log_root, exist_ok=True)

    # 复制主要运行文件
    copy_files_list = ['/usr/volume/WSSS/WSSS_PML_origin/train_cls_loc_jointly_merge_infer.py', 
                        '/usr/volume/WSSS/WSSS_PML_origin/network/resnet38_cls_ser_jointly_revised_seperatable.py',
                        '/usr/volume/WSSS/WSSS_PML_origin/tool/RoiPooling_Jointly.py',
                        '/usr/volume/WSSS/WSSS_PML_origin/tool/pyutils.py']
    for copy_file in copy_files_list: 
        shutil.copy(copy_file, log_root)

    #### train from imagenet params
    # args.session_name="from_imageNet"
    # args.weights="/usr/volume/WSSS/WSSS_PML_origin/weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params"
    # args.lr=0.1
    #
    # args.optimizer="poly"
    #### train from imagenet params

    model = getattr(importlib.import_module(args.network), 'Net')(args)

    # 所有tb的日志记录在saved_checkpointslog下
    tblogger = SummaryWriter(args.tblog_dir+'log')
    # .log日志文件
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

    infer_dataset = voc12.data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                scales=[1, 0.5, 1.5, 2.0],
                                                inter_transform=torchvision.transforms.Compose(
                                                    [np.asarray,
                                                    model.normalize,
                                                    imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)


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
        '''
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
        '''

        print(f"epoch{ep} start training. Now is:{time.ctime(time.time())}")
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
            loss = loss_cls + loss_patch  * args.patch_loss_weight
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
                      'Loss: %.4f' % (avg_meter.get('loss')),
                      'Loss_cls: %.4f:'%(avg_meter1.get('loss_cls')),
                      'Loss_patch: %.4f:' % (avg_meter2.get('loss_patch')),
                      'imps:%.3f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.6f' % (optimizer.param_groups[0]['lr']), flush=True)
                avg_meter.pop()

        print(f"epoch{ep} end!!!!!!!!!!!!!!!!!!! Now is:{time.ctime(time.time())}")
        if args.optimizer=='adam':
            optimizer.adam_turn_step()

        # model_saved_root=f"/usr/volume/WSSS/WSSS_PML_origin/{args.tblog_dir}/{args.session_name}/"
        # os.makedirs(model_saved_root, exist_ok=True)
        # model_saved_dir = os.path.join(model_saved_root, f"{ep}ep.pth")

        # torch.save(model.module.state_dict(),model_saved_dir)
        avg_meter.pop()
        
        loss_dict = {'loss': loss_list[-1]}
        tblogger.add_scalars('cls_loss', loss_dict, itr)
        tblogger.add_scalar('cls_lr', optimizer.param_groups[0]['lr'], itr)


        # =========================    evaluation    ===============================
        model.eval()
        # makedir stuff =====
        if args.out_cam_pred is not None:
            if os.path.exists(args.out_cam_pred):
                shutil.rmtree(args.out_cam_pred)
            if not os.path.exists(args.out_cam_pred):
                os.makedirs(args.out_cam_pred)
            for background_threshold in bg_thresh:
                os.makedirs(f"{args.out_cam_pred}/{background_threshold}", exist_ok=True)
        
        # 最后一个epoch时保存cam
        if ep==args.max_epoches-1 and args.out_cam is not None:
            if os.path.exists(args.out_cam):
                shutil.rmtree(args.out_cam)
            if not os.path.exists(args.out_cam):
                os.makedirs(args.out_cam)

        # 最后一个epoch时保存crf的结果
        if ep==args.max_epoches-1 and args.out_crf is not None:
            if os.path.exists(args.out_crf):
                shutil.rmtree(args.out_crf)
            for t in crf_alpha:
                folder = args.out_crf + ('_%.1f' % t)
                if not os.path.exists(folder):
                    os.makedirs(folder)
        # =====

        n_gpus = torch.cuda.device_count()
        for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
            img_name = img_name[0]; label = label[0]

            # if args.out_cam is not None:
            #     if os.path.exists(os.path.join(args.out_cam, img_name + '.npy')):
            #         continue

            img_path = voc12.data.get_img_path(img_name, args.voc12_root)
            orig_img = np.asarray(Image.open(img_path))
            orig_img_size = orig_img.shape[:2]

            def _work(i, img):
                with torch.no_grad():
                    with torch.cuda.device(i%n_gpus):
                        cam = model(img.cuda())
                        # print(cam)
                        cam = F.relu(cam, inplace=True)
                        cam = F.interpolate(cam, orig_img_size, mode='bilinear', align_corners=False)[0]   # [1, 20, 366, 500] to [20, 366, 500]
                        cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()  # label [20] to [20, 1, 1]
                        if i % 2 == 1:
                            cam = np.flip(cam, axis=-1)
                        return cam

            thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)),
                                                batch_size=8, prefetch_size=0, processes=args.infer_num_workers)

            cam_list = thread_pool.pop_results()

            # img_list中的8张图大小不一样，所以没办法在同一次前传过程中传入
            # model_out = model(torch.from_numpy(np.array(img_list)).cuda())
            # cams = F.relu(model_out, inplace=True)
            # cams = F.interpolate(cams, orig_img_size, mode='bilinear', align_corners=False)
            # # cams = F.interpolate(cams, orig_img_size, mode='bilinear', align_corners=False)[0]
            # cams = cams.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
            # cam_list = [np.flip(cam, axis=-1) if i%2 == 1 else cam for cam in enumerate(cams)]

            sum_cam = np.sum(cam_list, axis=0)
            norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)

            cam_dict = {}
            for i in range(20):
                if label[i] > 1e-5:
                    cam_dict[i] = norm_cam[i]

            # 最后一个epoch时才保存
            if ep==args.max_epoches-1 and args.out_cam is not None:
                # if not os.path.exists(args.out_cam):
                #     os.makedirs(args.out_cam)
                np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

            if args.out_cam_pred is not None:
                for background_threshold in bg_thresh:
                    bg_score = [np.ones_like(norm_cam[0])*background_threshold]
                    pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)
                    cv2.imwrite(os.path.join(f"{args.out_cam_pred}/{background_threshold}", img_name + '.png'), pred.astype(np.uint8))

            def _crf_with_alpha(cam_dict, alpha):
                v = np.array(list(cam_dict.values()))
                bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
                bgcam_score = np.concatenate((bg_score, v), axis=0)
                crf_score = imutils.crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

                n_crf_al = dict()

                n_crf_al[0] = crf_score[0]
                for i, key in enumerate(cam_dict.keys()):
                    n_crf_al[key+1] = crf_score[i+1]

                return n_crf_al

            # 最后一个epoch时才保存
            if ep==args.max_epoches-1 and args.out_crf is not None:
                for t in crf_alpha:
                    crf = _crf_with_alpha(cam_dict, t)
                    folder = args.out_crf + ('_%.1f'%t)
                    np.save(os.path.join(folder, img_name + '.npy'), crf)

            if iter%10==0:
                print(iter)
        
        result_saved_dir=os.path.join(f"{log_root}/log_txt/", f"{args.session_name}_{ep}")
        os.makedirs(f"{log_root}/log_txt/", exist_ok=True)
        for background_threshold in bg_thresh:
            if args.out_cam_pred is not None:
                print(f"background threshold is {background_threshold}")
                eval(f"/usr/volume/WSSS/WSSS_PML_origin/voc12/{phase}.txt", f"{args.out_cam_pred}/{background_threshold}", saved_txt=result_saved_dir, model_name=args.weights)

    print("Session finished:{}".format(time.ctime(time.time())))

    # np.save('loss.npy', loss_list)
    # np.save('validation_set_CAM_mIoU.npy', validation_set_CAM_mIoU)
