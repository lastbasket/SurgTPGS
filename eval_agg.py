"""
Evaluate segmentation using autoencoder-decoded features + CLIP relevancy from a HyperCLIP /
CAT-Seg checkpoint (codebook / finetuned text–image model).

Relevancy maps follow CAT-Seg ``CATSegPredictor`` (``forward_relevancy`` / ``_relevancy_positive_channel``):
per-pixel softmax over (positive vs negative phrases) with temperature, then the positive
channel probability is used for thresholding—same convention as ``eval_fine.py`` with
``preprocess_fine.OpenCLIPNetwork.get_relevancy``.
"""
from glob import glob
import os
import sys
import numpy as np
import torch
import argparse
import cv2
from tqdm import tqdm
from typing import Tuple
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
import torch.nn.functional as F
from scene import clip
from time import time

template_text = [
    'A photo of a {}',
    'A photo of a {} in the scene',
    'A photo of the {} in the scene',
    'This is a {} in the scene',
    'This is a photo of a {}',
    'This is a photo of a {} in the scene',
    '{}'
]

def cat_seg_relevancy_positive_channel(
    output: torch.Tensor,
    positive_id: int,
    num_pos: int,
    n_neg: int,
    temperature: float,
) -> torch.Tensor:
    """
    Match ``CATSegPredictor._relevancy_positive_channel``:
    for one class, compare positive similarity vs each negative in a 2-way softmax, pick worst negative, return (pos_prob, neg_prob) for that pair; caller uses channel 0 as score.
    """
    positive_vals = output[:, positive_id : positive_id + 1]
    negative_vals = output[:, num_pos :]
    repeated_pos = positive_vals.repeat(1, n_neg)
    sims = torch.stack((repeated_pos, negative_vals), dim=-1)
    softmax = torch.softmax(temperature * sims, dim=-1)
    best_id = softmax[..., 0].argmin(dim=1)
    gathered = torch.gather(
        softmax,
        1,
        best_id[..., None, None].expand(best_id.shape[0], n_neg, 2),
    )
    return gathered[:, 0, :]


class CatSegCodebookRelevancy(torch.nn.Module):
    """
    Text relevancy head matching ``CATSegPredictor`` when ``use_relevancy_head`` is True:
    ``embed @ [pos_embeds; neg_embeds].T`` then per-class positive-channel scores.
    """

    def __init__(
        self,
        clip_model: torch.nn.Module,
        tokenizer,
        negatives: Tuple[str, ...] = ("object", "things", "stuff", "texture"),
        clip_n_dims: int = 512,
        relevancy_temperature: float = 10.0,
    ):
        super().__init__()
        self.model = clip_model
        self.tokenizer = tokenizer
        self.negatives = negatives
        self.clip_n_dims = clip_n_dims
        self.relevancy_temperature = relevancy_temperature
        self.positives = ("",)
        self._encode_negatives()

    def _encode_negatives(self):
        with torch.no_grad():
            tok = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            neg = self.model.encode_text(tok)
            neg = neg / neg.norm(dim=-1, keepdim=True)
        self.register_buffer("neg_embeds", neg, persistent=False)

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            # tok = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            tok = self.tokenizer(self.positives).cuda()
            pos = self.model.encode_text(tok)
            pos = pos / pos.norm(dim=-1, keepdim=True)
        self.register_buffer("pos_embeds", pos, persistent=False)

    def get_relevancy(self, embed: torch.Tensor) -> torch.Tensor:
        """Same layout as ``preprocess_fine.OpenCLIPNetwork.get_relevancy`` / CAT-Seg."""
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)
        output = torch.mm(embed, p.T) # [409920, 512] @ [512, 40] -> [409920, 40]
        # print(output.shape) # [51840, 40]
        num_pos = len(self.positives)
        n_neg = len(self.negatives)
        # print(num_pos, n_neg) # 36, 4
        return output
    
    def get_relevancy_by_id(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        return cat_seg_relevancy_positive_channel(
            embed,
            positive_id,
            len(self.positives),
            len(self.negatives),
            self.relevancy_temperature,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--dataset_name", type=str, default="endovis_2018")
    parser.add_argument("--template_text", type=int, default=1)
    parser.add_argument("--vlm", type=str, default="clip", choices=["clip", "fine", "surg", "agg"])
    parser.add_argument("--gt_path", type=str, default="/data/langsplat/sofa/segmentations/")
    parser.add_argument("--ckpt_path", type=str, default="autoencoder/ckpt/CholecSeg8k/video01_0080/best_ckpt.pth")
    parser.add_argument(
        "--catseg_ckpt",
        type=str,
        required=True,
        help="HyperCLIP / Detectron2 checkpoint containing sem_seg_head.predictor.clip_model.* weights.",
    )
    parser.add_argument(
        "--relevancy_temperature",
        type=float,
        default=10.0,
        help="Softmax temperature in CAT-Seg relevancy (see CATSegPredictor.relevancy_temperature).",
    )
    parser.add_argument(
        "--relevancy_threshold",
        type=float,
        default=0.5,
        help="Used when --relevancy_threshold_mode fixed; also fallback when too few masked pixels.",
    )
    parser.add_argument(
        "--relevancy_threshold_mode",
        type=str,
        choices=["fixed", "quantile", "mean_std"],
        default="fixed",
        help="fixed: global --relevancy_threshold. quantile: thr = quantile(score|mask). "
        "mean_std: thr = mean + sigma*std within mask.",
    )
    parser.add_argument(
        "--relevancy_quantile",
        type=float,
        default=0.85,
        help="For quantile mode: keep pixels with score >= this quantile of masked scores (0–1).",
    )
    parser.add_argument(
        "--relevancy_sigma",
        type=float,
        default=1.0,
        help="For mean_std mode: threshold = mean + sigma * std (masked scores).",
    )

    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--encoder_dims", nargs="+", type=int, default=[256, 128, 64, 32, 3])
    parser.add_argument("--decoder_dims", nargs="+", type=int, default=[16, 32, 64, 128, 256, 256, 512])

    args = parser.parse_args()


    dataset_name = args.dataset_name
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    ckpt_path = args.ckpt_path
    print("Autoencoder ckpt:", ckpt_path)
    print("CAT-Seg / HyperCLIP ckpt:", args.catseg_ckpt)
    print("Load GT from:", args.gt_path)
    print(
        "Relevancy (CAT-Seg style) temperature:",
        args.relevancy_temperature,
        "| threshold mode:",
        args.relevancy_threshold_mode,
        "| fixed_thr:",
        args.relevancy_threshold,
    )

    if args.vlm == "fine":
        from preprocess_fine import OpenCLIPNetworkConfig
        from autoencoder.model import Autoencoder
    elif args.vlm == "agg":
        from preprocess_agg import OpenCLIPNetworkConfig
        from autoencoder.model import Autoencoder

    else:
        from autoencoder.model import Autoencoder
        from preprocess import OpenCLIPNetworkConfig

    if "cholecseg" in args.gt_path:
        target_text = [
            "Black Background",
            "Abdominal Wall",
            "Liver",
            "Gastrointestinal Tract",
            "Fat",
            "Grasper",
            "Connective Tissue",
            "Blood",
            "Cystic Duct",
            "L-hook Electrocautery",
            "Gallbladder",
            "Hepatic Vein",
            "Liver Ligament",
        ]

    elif "endovis_2018" in args.gt_path:
        target_text = [
            "background-tissue",
            "shaft",
            "clasper",
            "wrist",
            "kidney-parenchyma",
            "covered-kidney",
            "thread",
            "clamps",
            "suturing-needle",
            "suction-instrument",
            "small-intestine",
            "ultrasound-probe",
        ]

    elif "cadis" in args.gt_path:
        target_text = [
            "Pupil",
            "Surgical Tape",
            "Hand",
            "Eye Retractors",
            "Iris",
            "Skin",
            "Cornea",
            "Hydro. Cannula",
            "Visc. Cannula",
            "Cap. Cystotome",
            "Rycroft Cannula",
            "Bonn Forceps",
            "Primary Knife",
            "Ph. Handpiece",
            "Lens Injector",
            "I/A Handpiece",
            "Secondary Knife",
            "Micromanipulator",
            "I/A Handpiece Handle",
            "Cap. Forceps",
            "R. Cannula Handle",
            "Ph. Handpiece Handle",
            "Cap. Cystotome Handle",
            "Sec. Knife Handle",
            "Lens Injector Handle",
            "Suture Needle",
            "Needle Holder",
            "Charleux Cannula",
            "Primary Knife Handle",
            "Vitrectomy Handpiece",
            "Mendez Ring",
            "Marker",
            "Hydrosdissection Cannula Handle",
            "Troutman Forceps",
            "Cotton",
            "Iris Hooks",
        ]

    else:
        raise ValueError(f"Unknown dataset from gt_path: {args.gt_path}")

    positives = tuple(template_text[args.template_text].format(i) for i in target_text)

    OpenCLIPNetworkConfig.positives_gt = target_text
    OpenCLIPNetworkConfig.positives = positives

    data_dir = f"{args.output_path}/{args.dataset_name}/test/ours_3000"
    print("Data_dir:", data_dir)
    render_seg_path = os.path.join(data_dir, "render_seg_npy")
    mask_path = os.path.join(data_dir, "masks")

    render_seg_list = sorted(glob(os.path.join(render_seg_path, "*.npy")))
    mask_list = sorted(glob(os.path.join(mask_path, "*.png")))

    output_dir = os.path.join(data_dir, f"seg_separated_codebook_template_{args.template_text}")

    # Load CLIP from HyperCLIP (Detectron2) checkpoint — same path as CAT-Seg predictor's clip_model
    clip_model, _ = clip.load("ViT-B/16", pretrained=args.catseg_ckpt, device="cuda", jit=False, prompt_depth=0, prompt_length=0)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    CLIP_model = CatSegCodebookRelevancy(
        clip_model,
        clip.tokenize,
        negatives=("object", "things", "stuff", "texture"),
        clip_n_dims=512,
        relevancy_temperature=args.relevancy_temperature,
    )
    
    CLIP_model.set_positives(list(OpenCLIPNetworkConfig.positives))

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims, \
        use_agg=(args.vlm == "agg")).to("cuda:0")
    model.load_state_dict(checkpoint)
    model.eval()

    MIoU = []
    results = []
    id_avg_result = {}
    ssims = []
    psnrs = []
    lpipss = []

    for name in OpenCLIPNetworkConfig.positives_gt:
        id_avg_result[name] = []
        
    # gt_feature_path = "/mnt/data2_hdd/yiming/SurgTPGS/data/endovis_2018/seq_5_sub/language_features_agg"
    gt_feature_path = f"{args.gt_path.replace('/test_seg', '')}/language_features_agg"
    
    gt_feature_list = sorted(glob(os.path.join(gt_feature_path, "*.npy")))
    test_idxs = [i for i in range(len(gt_feature_list)) if (i-1) % 8 == 0]
    gt_feature_list = [gt_feature_list[i] for i in test_idxs]

    for idx, seg_path in tqdm(enumerate(render_seg_list), total=len(render_seg_list)):
        render_path = seg_path.replace("render_seg_npy", "renders").replace(".npy", ".png")
        gt_img_path = seg_path.replace("render_seg_npy", "gt").replace(".npy", ".png")
        
        gt_seg_path = seg_path.replace("render_seg_npy", "gt_seg_npy")
        
        gt_seg = torch.from_numpy(np.load(gt_seg_path)).cuda() # [480, 854, 3]
        
        gt_feature = torch.from_numpy(np.load(gt_feature_list[idx])).cuda() # [1, 512, 192, 192]
        gt_feature = F.interpolate(gt_feature, size=(gt_seg.shape[0], gt_seg.shape[1]), mode='bilinear', align_corners=False)
        gt_feature = gt_feature.permute(0, 2, 3, 1) # [1, 192, 192, 512]
        gt_feature = gt_feature.reshape(1, -1, 512)
        
        render = (torch.tensor(cv2.imread(render_path)).cuda().permute(2, 0, 1)) / 255.0
        gt = (torch.tensor(cv2.imread(gt_img_path)).cuda().permute(2, 0, 1)) / 255.0

        data = torch.from_numpy(np.load(seg_path)).cuda().unsqueeze(0)
        mpath = seg_path.replace("render_seg_npy", "masks").replace(".npy", ".png")
        mask = (torch.from_numpy(cv2.imread(mpath, -1)).cuda())[..., 0] / 255.0
        mask = mask.bool()


        psnrs.append(psnr(render*mask, gt*mask))
        ssims.append(ssim(render*mask, gt*mask))
        lpipss.append(lpips(render*mask, gt*mask, net_type="vgg"))

        relevancy_dir = os.path.join(output_dir, f"{idx}".zfill(5))
        os.makedirs(relevancy_dir, exist_ok=True)
        
        with torch.no_grad():
            lvl, h, w, _ = data.shape
            data = data.permute(1, 2, 0, 3)
            outputs = model.decode(data.reshape(-1, 3)) # HW, 3 -> HW, 512
            
            # outputs = model.decode(gt_seg.reshape(-1, 3))
            # outputs = gt_feature
            outputs = outputs.view(lvl, h, w, -1) # [1, 480, 854, 512]
            
        features = outputs.permute(1, 2, 0, 3).flatten(0, 2) # [409920, 512]
        
        # query_num = 500
        # start_time = time()
        # for _ in range(query_num):
        relevancy_embed = CLIP_model.get_relevancy(features)
        per_view_result = []
        scores = []
        for positive_id in range(len(OpenCLIPNetworkConfig.positives)):
            relevancy = CLIP_model.get_relevancy_by_id(relevancy_embed, positive_id) #.view(h, w, -1).permute(2, 0, 1)
            scores.append(relevancy[:, 0])
        scores = torch.stack(scores, dim=1) # [51840, 36]
        scores = scores.view(h, w, -1).permute(2, 0, 1) # [36, h, w]
        logits = torch.logit(scores.clamp(1e-6, 1.0 - 1e-6))
        logits = logits.sigmoid()
        sem_seg = logits.argmax(dim=0) # [h, w]
        # end_time = time()
        # print(f"Query time: {12*query_num/(end_time - start_time)} FPS")
        # # cv2.imwrite("sem_seg.png", (sem_seg.detach().cpu().numpy()*10).astype(np.uint8))
        # exit()
        for positive_id in range(len(OpenCLIPNetworkConfig.positives)):

            gt_mask_path = (
                f"{args.gt_path}/{str(idx).zfill(5)}/"
                f"{(OpenCLIPNetworkConfig.positives_gt[positive_id]).replace('/', '_')}.png"
            )
            gt_mask_path = gt_mask_path.replace("shaft", "instrument-shaft").replace("clasper", "instrument-clasper").replace("wrist", "instrument-wrist")
            gt_mask = torch.tensor(cv2.imread(gt_mask_path, -1)).to("cuda:0") / 255
            if gt_mask.sum() == 0:
                continue
            
            relevancy_mask = (sem_seg == positive_id)
            relevancy_mask = relevancy_mask * mask

            relevancy_mask_save = (relevancy_mask * 255).detach().cpu().numpy().astype(np.uint8)
            relevancy_path = os.path.join(relevancy_dir, f"{OpenCLIPNetworkConfig.positives[positive_id]}.png")
            cv2.imwrite(relevancy_path, relevancy_mask_save)

            intersection = torch.sum(torch.logical_and(relevancy_mask[None], gt_mask[None]))
            union = torch.sum(torch.logical_or(gt_mask[None], relevancy_mask[None]))
            miou = torch.sum(intersection) / torch.sum(union)
            per_view_result.append(
                {"class": OpenCLIPNetworkConfig.positives[positive_id], "MIoU": round(miou.item(), 2)}
            )

            MIoU.append(miou.data)
            id_avg_result[OpenCLIPNetworkConfig.positives_gt[positive_id]] += [miou.item()]

        results.append({"name": f"{idx}".zfill(5), "result": per_view_result})

    if len(MIoU) == 0:
        print("No valid GT masks evaluated; MIoU list empty.")
        total_avg = float("nan")
    else:
        total_avg = sum(MIoU) / len(MIoU)

    for name in id_avg_result:
        if len(id_avg_result[name]) == 0:
            id_avg_result[name] = 0
        else:
            id_avg_result[name] = np.array(id_avg_result[name]).mean()

    psnr_mean = torch.stack(psnrs).mean().item()
    ssim_mean = torch.stack(ssims).mean().item()
    lpips_mean = torch.stack(lpipss).mean().item()

    results_out = {
        "Results": results,
        "Total Avg Per Class": id_avg_result,
        "Total Average": round(float(total_avg), 2) if not np.isnan(total_avg.detach().cpu().numpy()) else None,
        "PSNR": psnr_mean,
        "SSIM": ssim_mean,
        "LPIPS": lpips_mean,
        "catseg_ckpt": args.catseg_ckpt,
        "relevancy_temperature": args.relevancy_temperature,
    }

    json_path = os.path.join(data_dir, f"result_codebook_template_{args.template_text}.json")
    with open(json_path, "w+") as fp:
        json.dump(results_out, fp, indent=4)
    print("mIoU average:", results_out["Total Average"])
    print("Wrote", json_path)
