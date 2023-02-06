## Datasets
Please download the datasets [here]() and extract them into root directory.

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
CUDA_VISIBLE_DEVICES=0 python train.py --data data/DBP15K --lang EN_FR
```

