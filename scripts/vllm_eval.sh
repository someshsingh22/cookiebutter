pip install awscli
bash /sensei-fs/users/someshs/scripts/aws.sh
bash /sensei-fs/users/someshs/scripts/git.sh
bash /sensei-fs/users/someshs/scripts/vars.sh

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
source ~/.bashrc

conda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm
pip install pandas pyarrow fastparquet chardet pandas
huggingface-cli login

aws s3 sync s3://crawldatafromgcp/somesh/KPITranslation/ckpt/composer/v3_hf/v3_3400/ ./v3_3400

