apt-get update
pip install -r requirements.txt
pip install loguru
pip install thop
pip install torch==1.11.0 torchvision==0.12.0
pip install opencv-contrib-python
python setup.py develop
apt-get install libgl1-mesa-glx -y
pip install wandb
