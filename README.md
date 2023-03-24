## Environment
```
python==3.6.13
apex==0.1
pytorch==1.7.1
torch_geometric==2.0.2
networkx==2.5.1
cuda==11.2
```
## Running
```
download glove.6B.zip from https://nlp.stanford.edu/projects/glove/
unzip glove.6B.zip into data/DBP15K/ (glove.6B.300d.txt will be used)
CUDA_VISIBLE_DEVICES=0 python train.py --data data/DBP15K --lang EN_FR_15K_V1
```

