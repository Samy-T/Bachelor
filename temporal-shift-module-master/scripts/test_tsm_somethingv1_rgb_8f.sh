# efficient setting: center crop and 1 clip
python test_models.py something \
    --weights=pretrained/TSM_something_RGB_resnet50_shift8_blockres_avg_segment8_e45.pth \
    --test_segments=8 --batch_size=72 -j 0 --test_crops=1

# accurate setting: full resolution and 2 clips (--twice sample)
python test_models.py something \
    --weights=pretrained/TSM_something_RGB_resnet50_shift8_blockres_avg_segment8_e45.pth \
    --test_segments=8 --batch_size=72 -j 0 --test_crops=3  --twice_sample