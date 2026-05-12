import os
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy
from scene import clip
import torch
import torchvision
from torch import nn
from scene.relevancy_aggregator import RelevancyAggregator
from torch.nn import functional as F
from einops import rearrange

# try:
#     import open_clip
# except ImportError:
#     assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"


@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    # clip_model_type: str = "ViT-B-16"
    clip_model_type: str = "ViT-B/16"
    clip_resolution: tuple = (384, 384) if clip_model_type == "ViT-B/16" else (336, 336)
    use_multi_scale_aggregation: bool = True
    
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config, pretrained_path=None):
        super().__init__()
        self.config = config
        self.use_multi_scale_aggregation = self.config.use_multi_scale_aggregation
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        # in 255 scale
        self.clip_pixel_mean = torch.tensor([122.7709383, 116.7460125, 104.09373615])
        self.clip_pixel_std = torch.tensor([68.5005327, 66.6321579, 70.3231630])
        # for OpenAI models
        model, _ = clip.load(self.config.clip_model_type,
                            pretrained=pretrained_path,  
                            device="cuda", jit=False, 
                            prompt_depth=0, 
                            prompt_length=0)
        
        print('Using CLIP model with CAT-Seg finetuned')
        model.eval()
        self.tokenizer = clip.tokenize
        self.model = model.to("cuda")
        
        self.feature_resolution = (24, 24)
        self.layer_indexes = [3, 7] if self.config.clip_model_type == "ViT-B/16" else [7, 15]
        self.layers = []
        for l in self.layer_indexes:
            self.model.visual.transformer.resblocks[l].register_forward_hook(
                lambda m, _, o: self.layers.append(o)
            )
            
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.positives]
                ).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, img_input, dense=False):
        if not self.use_multi_scale_aggregation:
            img_input = self.process(img_input)
        return self.model.encode_image(img_input, dense=dense)


def create(image_list, data_list, save_folder, clip_model, relevancy_aggregator):
    assert image_list is not None, "image_list must be provided to generate features"
    # the original sam segmentation maps are 

    for i, img in tqdm(enumerate(image_list), desc="Embedding images", leave=False):
        # pre-process the image
        img = img.float()
        mean = clip_model.clip_pixel_mean.to(dtype=img.dtype, device=img.device).view(3, 1, 1)
        std = clip_model.clip_pixel_std.to(dtype=img.dtype, device=img.device).view(3, 1, 1)
        img = (img - mean) / std
        img_embed = _embed_clip_sam_tiles(img.unsqueeze(0), clip_model, relevancy_aggregator)
        save_path = os.path.join(save_folder, data_list[i].split('.')[0] + '_agg.npy')
        np.save(save_path, img_embed.detach().cpu().numpy())
        

def _embed_clip_sam_tiles(image, clip_model, relevancy_aggregator):
    image = image.to("cuda", dtype=torch.float32)
    clip_images_resized = F.interpolate(image, size=clip_model.config.clip_resolution, mode='bicubic', align_corners=False, ) # [1, 3, 384, 384]
    clip_model.layers = []
    with torch.no_grad():
        clip_embed = clip_model.encode_image(clip_images_resized, dense=True) # should be [1, 577, 512]
        clip_features = rearrange(clip_embed[:, 1:, :], "b (h w) c->b c h w", h=clip_model.feature_resolution[0], w=clip_model.feature_resolution[1])
        
        image_features = clip_embed[:, 1:, :]
        # CLIP ViT features for guidance
        res3 = rearrange(image_features, "B (H W) C -> B C H W", H=clip_model.feature_resolution[0]) #[2, 512, 24, 24]
        res4 = rearrange(clip_model.layers[0][1:, :, :], "(H W) B C -> B C H W", H=clip_model.feature_resolution[0]) # [2, 768, 24, 24]
        res5 = rearrange(clip_model.layers[1][1:, :, :], "(H W) B C -> B C H W", H=clip_model.feature_resolution[0]) # [2, 768, 24, 24]
        features = {'res5': res5, 'res4': res4, 'res3': res3,}
        relevancy_features = relevancy_aggregator(clip_features, features) # [1, 512, 192, 192]
        
    relevancy_features = relevancy_features / relevancy_features.norm(dim=1, keepdim=True).clamp(min=1e-6)
    
    return relevancy_features


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--image_folder', type=str, required=True)
    
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_path', type=str, default="ckpts/sam_vit_h_4b8939.pth")
    parser.add_argument('--clip_ckpt_path', type=str, default="ckpts/model_final_cholecseg.pth")
    parser.add_argument('--aggregator_ckpt_path', type=str, default="ckpts/relevancy_aggregator_model_final.pth")
    
    
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    img_folder = os.path.join(dataset_path, args.image_folder)
    data_list = os.listdir(img_folder)
    data_list.sort()

    clip_model = OpenCLIPNetwork(OpenCLIPNetworkConfig, args.clip_ckpt_path)
    relevancy_aggregator = RelevancyAggregator(proj_dim=768, out_ch=512).to('cuda')
    agg_data = torch.load(args.aggregator_ckpt_path, map_location="cpu")
    relevancy_aggregator.load_state_dict(agg_data)
    relevancy_aggregator.eval()
    
    
    img_list = []
    WARNED = False
    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_w, orig_h = image.shape[1], image.shape[0]
        if args.resolution == -1:
            if orig_h > 1080:
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
            
        scale = float(global_down)
        resolution = (int( orig_w  / scale), int(orig_h / scale))
        
        image = cv2.resize(image, resolution)
        image = torch.from_numpy(image)
        img_list.append(image)
    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    imgs = torch.cat(images)

    save_folder = os.path.join(dataset_path, 'language_features_agg')
    os.makedirs(save_folder, exist_ok=True)
    create(imgs, data_list, save_folder, clip_model, relevancy_aggregator)