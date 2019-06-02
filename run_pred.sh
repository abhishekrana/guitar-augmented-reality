# apt update
# apt-get install -y libsm6 libxext6 libxrender-dev
# pip install opencv-python 
# pip install scikit-image
# pip install Augmentor
# pip install pudb

rm -rf output/*
# cp pred_image.jpg output/
# cp pred_corners.pkl output/
python3 prediction.py
