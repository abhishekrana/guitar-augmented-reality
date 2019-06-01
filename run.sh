# pip3 install pudb
# pip3 install Augmentor


### DATA AUGMENTATION - OFFLINE ###
## Train
# rm -rf output/*
# python3 data_aug_offline.py
# mkdir -p data/guitar/dataset_frames1_train_aug_v3/
# mv output/aug data/guitar/dataset_frames1_train_aug_v3/
# mv output/aug2/* data/guitar/dataset_frames1_train_aug_v3/

## Val
# rm -rf output/*
# python3 data_aug_offline.py
# mkdir -p data/guitar/dataset_frames1_val_aug_v3/
# mv output/aug data/guitar/dataset_frames1_val_aug_v3/
# mv output/aug2/* data/guitar/dataset_frames1_val_aug_v3/



# rm -rf output/*
python3 main.py

# cp -r data/guitar/dataset_frames1_train/* output/
# cp -r data/guitar/test/* output/

# rm -rf output/*
# python3 threshold.py

# rm -rf output/*
# python3 video_to_images.py

# rm -rf output/*
# python3 setup_data_annotated.py


# rm -rf output/*
# python3 fretboard.py

# rm -rf output/*
# mkdir -p output
# python3 homography.py

# python3 scripts/crop_image.py data/guitar/test/2019-05-28-085835_1.jpg 0 0 300 0
