#!/usr/bin/env bash



for name in 01_00080 01_15019 01_00240 12_15750 17_01803
do
        for level in 0 #1 #2 3
        do
            CASE_NAME="cholecseg_sub/video${name}_${level}"

            gt_folder="./data/cholecseg_sub/video${name}/test_seg"

            root_path="."

            python3 eval_agg.py \
              --dataset_name "${CASE_NAME}" \
              --output_path "${root_path}/output" \
              --encoder_dims 256 128 64 32 3 \
              --decoder_dims 16 32 64 128 256 256 512 \
              --gt_path "${gt_folder}" \
              --ckpt_path "autoencoder/ckpt/cholecseg_sub/video${name}/best_ckpt.pth" \
              --catseg_ckpt "ckpts/model_cholecseg_extract.pth" \
              --relevancy_threshold_mode fixed \
              --relevancy_threshold 0.5 \
              --level "${level}" \
              --vlm agg \
              --template_text 1
        done
done

for name in seq_5_sub #seq_9_sub
do
        for level in 0 #1 #2 3
        do
            #!/bin/bash
            CASE_NAME="endovis_2018/${name}_${level}"

            gt_folder="./data/endovis_2018/${name}/test_seg"

            root_path="."

            python3 eval_agg.py \
              --dataset_name "${CASE_NAME}" \
              --output_path "${root_path}/output" \
              --encoder_dims 256 128 64 32 3 \
              --decoder_dims 16 32 64 128 256 256 512 \
              --gt_path "${gt_folder}" \
              --ckpt_path "autoencoder/ckpt/endovis_2018/${name}/best_ckpt.pth" \
              --catseg_ckpt "ckpts/model_endovis_extract.pth" \
              --relevancy_threshold_mode fixed \
              --relevancy_threshold 0.5 \
              --level "${level}" \
              --vlm agg \
              --template_text 1
        done
done


for name in Video02 Video12 Video22; do
  for level in 0; do # 1 2 3
    CASE_NAME="cadisv2_test/${name}_${level}"
    gt_folder="./data/cadisv2_test/${name}/test_seg"
    root_path="."

    python3 eval_agg.py \
      --dataset_name "${CASE_NAME}" \
      --output_path "${root_path}/output" \
      --encoder_dims 256 128 64 32 3 \
      --decoder_dims 16 32 64 128 256 256 512 \
      --gt_path "${gt_folder}" \
      --ckpt_path "autoencoder/ckpt/cadisv2_test/${name}/best_ckpt.pth" \
      --catseg_ckpt "ckpts/model_cadis_extract.pth" \
      --relevancy_threshold_mode fixed \
      --relevancy_threshold 0.5 \
      --level "${level}" \
      --vlm agg \
      --template_text 1
  done
done
