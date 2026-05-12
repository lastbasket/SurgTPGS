import os
import numpy as np
import torch
import argparse
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--wo_agg', action='store_true', default=False)
    parser.add_argument('--model_type', type=str, default='mlp', choices=['mlp', 'convnext'])
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[256, 128, 64, 32, 3],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[16, 32, 64, 128, 256, 256, 512],
                    )
    parser.add_argument("--vlm", type=str, default = "CLIP")
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    dataset_path = args.dataset_path
    ckpt_path = f"ckpt/{dataset_name}/best_ckpt.pth"

    if args.vlm == "clip_fine" and args.wo_agg:
        data_dir = f"{dataset_path}/language_features_fine_wo_agg"
        output_dir = f"{dataset_path}/language_features_fine_wo_agg_dim3"
    elif args.vlm == "clip_fine":
        data_dir = f"{dataset_path}/language_features_fine"
        output_dir = f"{dataset_path}/language_features_fine_dim3"
    elif args.vlm == "clip":
        data_dir = f"{dataset_path}/language_features"
        output_dir = f"{dataset_path}/language_features_dim3"
    elif args.vlm == "agg":
        data_dir = f"{dataset_path}/language_features_agg"
        output_dir = f"{dataset_path}/language_features_agg_dim3"
    elif args.vlm == "surg":
        data_dir = f"{dataset_path}/language_features_surg"
        output_dir = f"{dataset_path}/language_features_surg_dim3"
    elif args.vlm == "surgB":
        data_dir = f"{dataset_path}/language_features_surgB"
        output_dir = f"{dataset_path}/language_features_surgB_dim3"
    else:
        data_dir = f"{dataset_path}/language_features"
        output_dir = f"{dataset_path}/language_features_dim3"
        
    print('Dataset Name:', dataset_name)
    print('Features Path:', data_dir)
    print('Use Checkpoint:', ckpt_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if (args.vlm != "agg") and (args.vlm != "surg") and (args.vlm != "surgB") and (args.vlm != "clip"):
        # copy the segmentation map
        for filename in os.listdir(data_dir):
            if filename.endswith("_s.npy"):
                source_path = os.path.join(data_dir, filename)
                target_path = os.path.join(output_dir, filename)
                shutil.copy(source_path, target_path)

    checkpoint = torch.load(ckpt_path)
    train_dataset = Autoencoder_dataset(data_dir, args.model_type)

    test_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=16,
        shuffle=False, 
        num_workers=16, 
        drop_last=False   
    )

    if args.model_type == 'convnext':
        from model_convnext import Autoencoder
        model = Autoencoder().to("cuda:0")
    else:
        from model import Autoencoder
        model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims, use_agg=(args.vlm == "agg")).to("cuda:0")

    model.load_state_dict(checkpoint)
    model.eval()

    for idx, feature in tqdm(enumerate(test_loader)):
        data = feature.to("cuda:0")
        with torch.no_grad():
            data_len = data.shape[0]
            data = data.reshape(-1, 512)
            
            outputs = model.encode(data).to("cpu").numpy()  
            if args.vlm == "agg" or args.vlm == "surg" or args.vlm == "clip":
                outputs = outputs.reshape(data_len, 192, 192, 3) # [1, 192*192, 512] -> [1, 192, 192, 512]
            elif args.vlm == "surgB":
                outputs = outputs.reshape(data_len, 192, 192, 3) # [1, 192*192, 512] -> [1, 192, 192, 512]
        if idx == 0:
            features = outputs
        else:
            features = np.concatenate([features, outputs], axis=0)

    os.makedirs(output_dir, exist_ok=True)
    start = 0
    
    for k,v in train_dataset.data_dic.items():
        path = os.path.join(output_dir, k)
        np.save(path, features[start:start+v])
        start += v
