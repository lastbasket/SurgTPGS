for name in  01_00080 01_15019  01_00240 12_15750 #17_01803       
do
    python render.py --model_path output/cholecseg_sub/video${name}_0  --skip_train \
    --skip_video --configs arguments/endonerf/default.py
done

for name in seq_5_sub seq_9_sub 
do
    python render.py --model_path output/endovis_2018/${name}_0  --skip_train \
    --skip_video --configs arguments/endonerf/default.py
done

for name in Video02 Video12 Video22
do
    python render.py --model_path output/cadisv2_test/${name}_0  --skip_train \
    --skip_video --configs arguments/endonerf/default.py
done