# apt update
# apt-get install -y libsm6 libxext6 libxrender-dev
# pip install opencv-python 
# pip install scikit-image
# pip install Augmentor
# pip install pudb



rm -rf output/frames_pred/*
# cp -r data/guitar/test_4/*.jpg output/frames_pred/
python3 prediction.py
