# train the autoencoder
cd autoencoder
for name in 01_00080 01_00240 01_15019 12_15750 17_01803
do
    python train.py --dataset_name cholecseg_sub/video${name} --dataset_path ../data/cholecseg_sub/video${name} \
    --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --vlm clip_fine 

    # get the 3-dims language feature of the scene
    python test.py --dataset_name cholecseg_sub/video${name} --dataset_path ../data/cholecseg_sub/video${name} --vlm clip_fine
done

for name in seq_5_sub seq_9_sub
do
    python train.py --dataset_name endovis_2018/${name} --dataset_path ../data/endovis_2018/${name} \
    --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --vlm clip_fine 

    # get the 3-dims language feature of the scene
    python test.py --dataset_name endovis_2018/${name} --dataset_path ../data/endovis_2018/${name} --vlm clip_fine
done