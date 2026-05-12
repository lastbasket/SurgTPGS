for data_dir in video01_00080 video01_00240 video01_15019 video12_15750 video17_01803   
do
    python preprocess_agg.py --dataset_path ./data/cholecseg_sub/${data_dir} --image_folder images --clip_ckpt_path ckpts/model_cholecseg_extract.pth --aggregator_ckpt_path ckpts/model_cholecseg_agg.pth
done

for data_dir in seq_5_sub seq_9_sub 
do
    python preprocess_agg.py --dataset_path ./data/endovis_2018/${data_dir} --image_folder images --clip_ckpt_path ckpts/model_endovis_extract.pth --aggregator_ckpt_path ckpts/model_endovis_agg.pth
done

for data_dir in Video02 Video12 Video22
do
    python preprocess_agg.py --dataset_path ./data/cadisv2_test/${data_dir} --image_folder images --clip_ckpt_path ckpts/model_cadis_extract.pth --aggregator_ckpt_path ckpts/model_cadis_agg.pth
done