import cv2
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
import voc12.data
import scipy.misc
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, pyutils
import argparse
from PIL import Image
import torch.nn.functional as F
import os.path
import shutil

from evaluation import eval

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

if __name__ == '__main__':
    torch.set_num_threads(16)
    phase = "val"
    bg_thresh=[0.21,0.22,0.23,0.24,0.243,0.245,0.248,0.25]
    # bg_thresh=[0.20,0.205,0.21,0.22,0.23,0.24,0.243,0.245,0.248,0.25,0.253,0.255,0.258,0.26,0.263,0.265,0.268,0.27,0.273,0.275,0.278,0.28]
    # bg_thresh=[0.23]

    parser = argparse.ArgumentParser()
    # parser.add_argument("--weights", default="/usr/volume/WSSS/weights_released/res38_cls.pth", type=str)
    # parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--weights", default="/usr/volume/WSSS/WSSS_PML_origin/saved_checkpoints/final_poly_bs_24_ep3/2ep.pth", type=str)
    parser.add_argument("--network", default="network.resnet38_cls_ser_jointly_revised_seperatable", type=str)

    # parser.add_argument("--infer_list", default=f"/usr/volume/WSSS/WSSS_PML_origin/voc12/train_aug.txt", type=str)
    # parser.add_argument("--infer_list", default=f"/usr/volume/WSSS/WSSS_PML_origin/voc12/{phase}_voc12.txt", type=str)
    # parser.add_argument("--infer_list", default=f"/usr/volume/WSSS/psa/voc12/val.txt", type=str)
    parser.add_argument("--infer_list", default=f"/usr/volume/WSSS/WSSS_PML_origin/voc12/val_voc12.txt", type=str)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--voc12_root", default='/usr/volume/WSSS/VOCdevkit/VOC2012', type=str)

    parser.add_argument("--out_cam", default="./out_cam_val", type=str)
    parser.add_argument("--out_crf", default="./out_crf", type=str)
    # parser.add_argument("--out_cam", default=None, type=str)
    # parser.add_argument("--out_crf", default=None, type=str)

    parser.add_argument("--out_cam_pred", default="./out_cam_pred_val", type=str)
    parser.add_argument("--log_infer_cls", default=f"/usr/volume/WSSS/WSSS_PML_origin/log_CAM_{phase}.txt", type=str)
    parser.add_argument("--bg_threshold", default=0.245, type=float)

    args = parser.parse_args()
    crf_alpha = [4, 16, 24, 28, 32]

    # background_threshold = args.bg_threshold

    # makedir stuff =====
    if args.out_cam_pred is not None:
        if os.path.exists(args.out_cam_pred):
            shutil.rmtree(args.out_cam_pred)

    if args.out_cam_pred is not None:
        if not os.path.exists(args.out_cam_pred):
            os.makedirs(args.out_cam_pred)
        for background_threshold in bg_thresh:
            os.makedirs(f"{args.out_cam_pred}/{background_threshold}", exist_ok=True)

    if args.out_cam is not None:
        if not os.path.exists(args.out_cam):
            os.makedirs(args.out_cam)

    if args.out_crf is not None:
        for t in crf_alpha:
            folder = args.out_crf + ('_%.1f' % t)
            if not os.path.exists(folder):
                os.makedirs(folder)
    # =====


    model = getattr(importlib.import_module(args.network), 'Net')()
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    infer_dataset = voc12.data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                   scales=[1, 0.5, 1.5, 2.0],
                                                   inter_transform=torchvision.transforms.Compose(
                                                       [np.asarray,
                                                        model.normalize,
                                                        imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))

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
                    cam = model_replicas[i%n_gpus](img.cuda())
                    # print(cam)
                    cam = F.relu(cam, inplace=True)
                    cam = F.interpolate(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    return cam

        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)),
                                            batch_size=8, prefetch_size=0, processes=args.num_workers)

        cam_list = thread_pool.pop_results()

        sum_cam = np.sum(cam_list, axis=0)
        norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)

        cam_dict = {}
        for i in range(20):
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i]

        # if args.out_cam is not None:
        #     # if not os.path.exists(args.out_cam):
        #     #     os.makedirs(args.out_cam)
        #     np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

        if args.out_cam_pred is not None:
            for background_threshold in bg_thresh:
                bg_score = [np.ones_like(norm_cam[0])*background_threshold]
                pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)
                # import pdb; pdb.set_trace()
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

        if args.out_crf is not None:
            for t in crf_alpha:
                crf = _crf_with_alpha(cam_dict, t)
                folder = args.out_crf + ('_%.1f'%t)
                # if not os.path.exists(folder):
                #     os.makedirs(folder)
                np.save(os.path.join(folder, img_name + '.npy'), crf)

        if iter%5==0:
            print(iter)
            
    for background_threshold in bg_thresh:
        if args.out_cam_pred is not None:
            print(f"background threshold is {background_threshold}")
            eval(f"/usr/volume/WSSS/WSSS_PML_origin/voc12/{phase}.txt", f"{args.out_cam_pred}/{background_threshold}", saved_txt=args.log_infer_cls, model_name=args.weights)



