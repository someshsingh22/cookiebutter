bash /sensei-fs/users/someshs/scripts/aws.sh
bash /sensei-fs/users/someshs/scripts/git.sh
bash /sensei-fs/users/someshs/scripts/vars.sh

aws s3 cp ###

cp -r /sensei-fs/users/someshs/llm-foundry-llama/ ./
pip install -e ./llm-foundry-llama

