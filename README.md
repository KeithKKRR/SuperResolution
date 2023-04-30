# SuperResolution




conda create -n sr python=3.8 -y

conda activate sr

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

python -m pip install --upgrade pip

pip install pillow opencv-python tqdm