# Maya Setup

## Build LC0

```bash
bash /sensei-fs/users/someshs/scripts/aws.sh
bash /sensei-fs/users/someshs/scripts/git.sh
bash /sensei-fs/users/someshs/scripts/vars.sh
git clone https://github.com/someshsingh22/lc0.git
cd lc0
sudo apt install libgtest-dev -y
pip install ninja python-chess meson
./build.sh
echo 'export PATH=$PATH:/home/user/lc0/build/release' >> ~/.bashrc && source ~/.bashrc
pip install onnx onnxruntime onnx2torch
mkdir weights
mkdir evals
wget https://storage.lczero.org/files/networks-contrib/big-transformers/BT4-1024x15x32h-swa-6147500.pb.gz -P weights
lc0 leela2onnx --input=weights/BT4-1024x15x32h-swa-6147500.pb.gz --output=weights/BT4-1024x15x32h-swa-6147500.onnx
```

### For running with lc0 build

```bash
bash /sensei-fs/users/someshs/scripts/aws.sh
bash /sensei-fs/users/someshs/scripts/git.sh
bash /sensei-fs/users/someshs/scripts/vars.sh
git clone git@github.com:someshsingh22/rlqa.git
cd rlqa
rm -rf lc0
git clone https://github.com/someshsingh22/lc0.git
cd lc0
sudo apt install libgtest-dev -y
pip install ninja python-chess meson
./build.sh
echo 'export PATH=$PATH:/home/user/rlqa/lc0/build/release' >> ~/.bashrc && source ~/.bashrc
pip install onnx onnxruntime onnx2torch
mkdir weights
mkdir evals
wget https://storage.lczero.org/files/networks-contrib/big-transformers/BT4-1024x15x32h-swa-6147500.pb.gz -P weights
lc0 leela2onnx --input=weights/BT4-1024x15x32h-swa-6147500.pb.gz --output=weights/BT4-1024x15x32h-swa-6147500.onnx
cd /home/user/rlqa
```

`## Simulating Evals


```python
['cudnn-auto', 'cudnn', 'cudnn-fp16', 'cuda-auto', 'cuda', 'cuda-fp16', 'blas', 'eigen', 'trivial', 'random', 'check', 'recordreplay', 'roundrobin', 'multiplexing', 'demux']
w = Weights('BT4-1024x15x32h-swa-6147500.pb.gz')

```
`
## Get Projection

```python
import onnx
import torch
from onnx2torch import convert
import re




def forward_hook(module, input, output):
'''
Forward hook to save encoder outputs as a torch dict of tensors
'''


onnx_model_path =  'weights/BT4-1024x15x32h-swa-6147500.onnx'
onnx_model = convert(onnx_model_path)

encoder_layer_format = r"encoder[\d]+/ln2"

layers = [layer for layer in onnx_model.layers if re.match(encoder_layer_format, layer.name)]
```