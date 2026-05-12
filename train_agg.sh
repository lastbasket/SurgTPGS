for name in 01_00080 01_00240 01_15019 12_15750 #17_01803     
do
    for level in 0 #1 # 2 3
    do
        python train.py -s data/cholecseg_sub/video${name} --expname cholecseg_sub/video${name}_${level} \
        --configs arguments/endonerf/default.py --feature_level ${level} --vlm clip_fine
    done
done


for name in  seq_5_sub seq_9_sub
do
    for level in 0 #1 # 2 3
    do
        python train.py -s data/endovis_2018/${name} --expname endovis_2018/${name}_${level} \
        --configs arguments/endonerf/default.py --feature_level ${level} --vlm clip_fine
    done
done


for name in  Video02 Video12 Video22
do
    for level in 0 #1 # 2 3
    do
        python train.py -s data/cadisv2_test/${name} --expname cadisv2_test/${name}_${level} \
        --configs arguments/endonerf/default.py --feature_level ${level} --vlm clip_fine
    done
done