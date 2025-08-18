for name in 01_00080 01_00240 01_15019 12_15750 17_01803  
do
        for level in 0 #1 #2 3
        do
                CASE_NAME="cholecseg_sub/video${name}_${level}"

                gt_folder="./data/cholecseg_sub/video${name}/test_seg"

                root_path="."

                python eval_fine.py \
                        --dataset_name $CASE_NAME \
                        --output_path ${root_path}/output \
                        --encoder_dims 256 128 64 32 3 \
                        --decoder_dims 16 32 64 128 256 256 512 \
                        --gt_path ${gt_folder} \
                        --ckpt_path autoencoder/ckpt/cholecseg_sub/video${name}/best_ckpt.pth \
                        --clip_ckpt_path ckpts/model_final_cholecseg.pth \
                        --level ${level} \
                        --vlm fine
        done
done

for name in seq_5_sub seq_9_sub
do
        for level in 0 #1 #2 3
        do
                #!/bin/bash
                CASE_NAME="endovis_2018/${name}_${level}"

                gt_folder="./data/endovis_2018/${name}/test_seg"

                root_path="."

                python eval_fine.py \
                        --dataset_name $CASE_NAME \
                        --output_path ${root_path}/output \
                        --encoder_dims 256 128 64 32 3 \
                        --decoder_dims 16 32 64 128 256 256 512 \
                        --gt_path ${gt_folder} \
                        --ckpt_path autoencoder/ckpt/endovis_2018/${name}/best_ckpt.pth \
                        --clip_ckpt_path ckpts/model_final_endovis.pth \
                        --level ${level} \
                        --vlm fine
        done
done