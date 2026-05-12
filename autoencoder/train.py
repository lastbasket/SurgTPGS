import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset
from torch.utils.tensorboard import SummaryWriter
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR


torch.autograd.set_detect_anomaly(False)

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=2)
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
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument("--vlm", type=str, default = "CLIP")
    parser.add_argument('--min_lr', type=float, default=0)
    args = parser.parse_args()
    dataset_path = args.dataset_path
    num_epochs = args.num_epochs
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    if args.vlm == "clip_fine" and args.wo_agg:
        data_dir = f"{dataset_path}/language_features_fine_wo_agg"
    elif args.vlm == "clip_fine":
        data_dir = f"{dataset_path}/language_features_fine"
    elif args.vlm == "clip":
        data_dir = f"{dataset_path}/language_features"
    elif args.vlm == "agg":
        data_dir = f"{dataset_path}/language_features_agg"
    elif args.vlm == "surg":
        data_dir = f"{dataset_path}/language_features_surg"
    elif args.vlm == "surgB":
        data_dir = f"{dataset_path}/language_features_surgB"
    else:
        data_dir = f"{dataset_path}/language_features"
    print('Loading VLM features from:', data_dir)
    os.makedirs(f'ckpt/{args.dataset_name}', exist_ok=True)

    train_dataset = Autoencoder_dataset(data_dir, args.model_type)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False  
    )
    
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims

    if args.model_type == 'convnext':
        from model_convnext import Autoencoder
        model = Autoencoder().to("cuda:0")
    else:
        from model import Autoencoder
        model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims, use_agg=(args.vlm == "agg")).to("cuda:0")

    print("lr:", args.lr)
    print("batch_size:", args.batch_size)
    print("num_workers:", args.num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=args.min_lr)
    logdir = f'ckpt/{args.dataset_name}'
    tb_writer = SummaryWriter(logdir)
    print("logdir:", logdir)

    best_eval_loss = 100.0
    best_epoch = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for idx, feature in enumerate(train_loader):
            data = feature.to("cuda:0")
            optimizer.zero_grad(set_to_none=True)
            data = data.reshape(-1, 512)
            outputs = model(data)
            
            l2loss = l2_loss(outputs, data) 
            cosloss = cos_loss(outputs, data)
            loss = l2loss + cosloss * 0.001

            loss.backward()
            optimizer.step()

            global_iter = epoch * len(train_loader) + idx
            if idx % args.log_interval == 0:
                tb_writer.add_scalar('train_loss/l2_loss', l2loss.item()/args.batch_size, global_iter)
                tb_writer.add_scalar('train_loss/cos_loss', cosloss.item()/args.batch_size, global_iter)
                tb_writer.add_scalar('train_loss/total_loss', loss.item()/args.batch_size, global_iter)
                tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_iter)

        scheduler.step()

        if epoch > 95:
            eval_loss = 0.0
            model.eval()
            for idx, feature in enumerate(test_loader):
                data = feature.to("cuda:0", non_blocking=True)
                with torch.no_grad():
                    data = data.reshape(-1, 512)
                    outputs = model(data) 
                    loss = l2_loss(outputs, data) + cos_loss(outputs, data)*0.001
                eval_loss += loss.item()
            eval_loss = eval_loss / len(train_dataset)
            print("eval_loss:{:.8f}".format(eval_loss))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                torch.save(model.state_dict(), f'ckpt/{args.dataset_name}/best_ckpt.pth')
                
            if epoch % 10 == 0:
                torch.save(model.state_dict(), f'ckpt/{args.dataset_name}/{epoch}_ckpt.pth')
            
    print(f"best_epoch: {best_epoch}")
    print("best_loss: {:.8f}".format(best_eval_loss))