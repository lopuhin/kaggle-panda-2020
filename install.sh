set -ev

# Example setup, using pytorch/pytorch:1.5-cuda10.1-cudnn7-devel image as a base
# scp -P XXX install.sh root@ssh4.vast.ai:~/
# scp -r -P XXX ~/.kaggle/ root@ssh4.vast.ai:~/

apt update
apt install -y libturbojpeg vim git unzip libglib2.0-0

git clone git@github.com:lopuhin/kaggle-panda-2020.git

pip install --no-cache pip -U
cat kaggle-panda-2020/requirements.txt | grep -v 'inplace-abn' > r.txt
pip install --no-cache --pre -r r.txt -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html
rm r.txt
pip install --no-cache inplace-abn==1.0.12
pip install --no-cache kaggle

pip install -e kaggle-panda-2020

mkdir kaggle-panda-2020/data
cd kaggle-panda-2020/data
kaggle datasets download -d lopuhin/panda-2020-level-1-2
unzip panda-2020-level-1-2.zip
rm panda-2020-level-1-2.zip
mv train_images extracted
mv extracted/train_images .
rm -r extracted
cd ..
