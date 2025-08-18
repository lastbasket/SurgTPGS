for data_dir in video01_00080 video01_00240 video01_15019 video12_15750 video17_01803   
do
    python preprocess_fine.py --dataset_path ./data/cholecseg_sub/${data_dir} --image_folder images --clip_ckpt_path ckpts/model_final_cholecseg.pth
done

for data_dir in seq_5_sub seq_9_sub 
do
    python preprocess_fine.py --dataset_path ./data/endovis_2018/${data_dir} --image_folder images --clip_ckpt_path ckpts/model_final_endovis.pth
done
