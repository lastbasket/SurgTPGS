from glob import glob
import os
import numpy as np
import torch
import argparse
import shutil
import cv2
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Tuple, Type
from PIL import Image
from utils.image_utils import psnr
from utils.loss_utils import ssim
# from metrics import cal_lpips
from lpipsPyTorch import lpips
import json
from time import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument('--dataset_name', type=str, default='endovis_2018')
    parser.add_argument('--vlm', type=str, default='clip')
    parser.add_argument('--gt_path', type=str, default='/data/langsplat/sofa/segmentations/')
    parser.add_argument('--ckpt_path', type=str, default='autoencoder/ckpt/CholecSeg8k/video01_0080/best_ckpt.pth')
    parser.add_argument('--clip_ckpt_path', type=str, default='autoencoder/ckpt/CholecSeg8k/video01_0080/best_ckpt.pth')
    
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--encoder_dims',
    nargs='+',
    type=int,
    default=[256, 128, 64, 32, 3],
    )
    parser.add_argument('--decoder_dims',
    nargs='+',
    type=int,
    default=[16, 32, 64, 128, 256, 256, 512],
    )
    args = parser.parse_args()
    dataset_name = args.dataset_name
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    ckpt_path = args.ckpt_path
    print("This is  ckpt_path:{}".format(ckpt_path))
    print("Load GT from:", args.gt_path)    

    if args.vlm == 'fine':
        print("Use Finetune CLIP")
        from preprocess_fine import OpenCLIPNetwork, OpenCLIPNetworkConfig
        from autoencoder.model import Autoencoder
        
    elif args.vlm == 'surg':
        print("Use SurgVLP")
        from preprocess_surg import OpenCLIPNetwork, OpenCLIPNetworkConfig
        from autoencoder_surg.model import Autoencoder
        decoder_hidden_dims[-1] = 768
        
    else:
        print("Use CLIP")
        from autoencoder.model import Autoencoder
        from preprocess import OpenCLIPNetwork, OpenCLIPNetworkConfig
            
    if "cholecseg" in args.gt_path:
        target_text = ["Black Background", "Abdominal Wall", "Liver", "Gastrointestinal Tract", \
            "Fat", "Grasper", "Connective Tissue", "Blood", "Cystic Duct", "L-hook Electrocautery", \
                "Gallbladder", "Hepatic Vein", "Liver Ligament"]
        
        
    elif "endovis_2018" in args.gt_path:
        target_text = ["background-tissue", "instrument-shaft", "instrument-clasper", \
            "instrument-wrist", "kidney-parenchyma", "covered-kidney", "thread", "clamps", \
                "suturing-needle", "suction-instrument", "small-intestine", "ultrasound-probe"]
        
        
    positives = tuple('A photo of a {} in the scene'.format(i) for i in target_text)

    OpenCLIPNetworkConfig.positives_gt = target_text
    OpenCLIPNetworkConfig.positives = positives

    data_dir = f'{args.output_path}/{args.dataset_name}/test/ours_3000'
    print('Data_dir:', data_dir)
    render_seg_path = os.path.join(data_dir, 'render_seg_npy')
    mask_path = os.path.join(data_dir, 'masks')

    render_seg_list = sorted(glob(os.path.join(render_seg_path, "*.npy")))
    
    mask_list = sorted(glob(os.path.join(mask_path, "*.png")))
    
    output_dir = os.path.join(data_dir, 'seg_separated')

    if "cholecseg" in args.gt_path and args.vlm == 'fine':
        CLIP_model = OpenCLIPNetwork(OpenCLIPNetworkConfig, args.clip_ckpt_path)
    elif "endovis_2018" in args.gt_path and args.vlm == 'fine':
        CLIP_model = OpenCLIPNetwork(OpenCLIPNetworkConfig, args.clip_ckpt_path)
    else:
        CLIP_model = OpenCLIPNetwork(OpenCLIPNetworkConfig)

    checkpoint = torch.load(ckpt_path)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

    model.load_state_dict(checkpoint)
    model.eval()

    relevancy_masks = []
    relevancys = []
    gt_masks = []
    
    MIoU = []
    ACC = []
    results = []
    id_avg_result = {}
    ssims = []
    psnrs = []
    lpipss = []
    
    for name in OpenCLIPNetworkConfig.positives_gt:
        id_avg_result[name] = []
        
    for idx, feature in tqdm(enumerate(render_seg_list)): # total num of features
        seg_path = render_seg_list[idx]
        render_path = seg_path.replace('render_seg_npy', 'renders').replace('.npy', '.png')
        gt_path = seg_path.replace('render_seg_npy', 'gt').replace('.npy', '.png')
        
        render = (torch.tensor(cv2.imread(render_path)).cuda().permute(2,0,1))/255.0
        gt = (torch.tensor(cv2.imread(gt_path)).cuda().permute(2,0,1))/255.0
        
        psnrs.append(psnr(render, gt))
        ssims.append(ssim(render, gt))
        lpipss.append(lpips(render, gt, net_type='vgg'))
        data = torch.from_numpy(np.load(render_seg_list[idx])).cuda().unsqueeze(0)
        mask_path = seg_path.replace('render_seg_npy', 'masks').replace('.npy', '.png')
        mask = (torch.from_numpy(cv2.imread(mask_path, -1)).cuda())[..., 0]/255.0
        mask = mask.bool()
        
        data_path = feature[1][0]        
        data_gt_idx = os.path.basename(data_path).split(".npy")[0]
        # for time measure
        # time1 = time()
        
        # for i in range(500):
        with torch.no_grad():
            lvl, h, w, _ = data.shape
            data = data.permute(1, 2, 0, 3)
            outputs = model.decode(data.reshape(-1, 3)) # [409920, 512]
            # outputs = model.decode(gt_fea_dim3) # [409920, 512]
            outputs = outputs.view(lvl, h, w, -1) # [1, 480, 854, 512]
            # outputs_gt = outputs_gt.view(lvl, h, w, -1) # [1, 480, 854, 512]
            # outputs = outputs_gt
        features = outputs.permute(1, 2, 0, 3).flatten(0, 2)
        CLIP_model.set_positives(list(OpenCLIPNetworkConfig.positives)) # [0]
        
        # for time measure
        # relevancy = CLIP_model.get_relevancy(features, 0)#.view(h, w, -1).permute(2,0,1)
        # time2=time()
        # print("FPS:",(500)/(time2-time1))
        relevancy_dir = os.path.join(output_dir, f"{idx}".zfill(5))
        os.makedirs(relevancy_dir, exist_ok=True)
        
        per_view_result = []
        for positive_id in range(len(OpenCLIPNetworkConfig.positives)):
            if positive_id == 0:
                continue
            
            gt_mask_path = f'{args.gt_path}/{str(idx).zfill(5)}/{OpenCLIPNetworkConfig.positives_gt[positive_id]}.png'
            # print(gt_mask_path)
            gt_mask = torch.tensor(cv2.imread(gt_mask_path, -1)).to("cuda:0")/255
            if gt_mask.sum() == 0:
                continue
            
            relevancy = CLIP_model.get_relevancy(features, positive_id).view(h, w, -1).permute(2,0,1)
            relevancy_mask = torch.zeros_like(relevancy[0]).to(relevancy.device)
            relevancy_mask[relevancy[0] > 0.4] = 1
            relevancy_mask = relevancy_mask * mask
            # print(relevancy_mask.device)
            relevancy_mask_save = (relevancy_mask*255).detach().cpu().numpy().astype(np.uint8)
            relevancy_path = os.path.join(relevancy_dir, f"{OpenCLIPNetworkConfig.positives[positive_id]}.png")
            cv2.imwrite(relevancy_path, relevancy_mask_save)
            
            intersection = torch.sum(torch.logical_and(relevancy_mask[None], gt_mask[None]))
            union = torch.sum(torch.logical_or(gt_mask[None], relevancy_mask[None]))
            miou = torch.sum(intersection) / torch.sum(union)
            per_view_result.append({'class':OpenCLIPNetworkConfig.positives[positive_id],
                                    'MIoU':round(miou.item(), 2)})
            
            # if "choleseg" in args.gt_path and positive_id == 5:
            #     grasper_list.append(miou.data)
            MIoU.append(miou.data)
            id_avg_result[OpenCLIPNetworkConfig.positives_gt[positive_id]] += [miou.item()]
            
            
        results.append({'name': f"{idx}".zfill(5), 'result':per_view_result})

    total_avg = sum(MIoU)/len(MIoU)
    
    for name in id_avg_result:
        if len(id_avg_result[name]) == 0:
            id_avg_result[name] = 0
        else:    
            id_avg_result[name] = np.array(id_avg_result[name]).mean()
            
    
    psnr_mean = torch.stack(psnrs).mean().item()
    ssim_mean = torch.stack(ssims).mean().item()
    lpips_mean = torch.stack(lpipss).mean().item()
    
    results = {'Results': results, 'Total Avg Per Class': id_avg_result, 
               'Total Average': round(total_avg.item(), 2),
               'PSNR':psnr_mean, 'SSIM':ssim_mean, 'LPIPS':lpips_mean}
    
    json_path = os.path.join(data_dir, 'result.json')
    with open(json_path, 'w+') as fp:
        j = json.dump(results, fp, indent=4)
    print(round(total_avg.item(), 2))