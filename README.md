# SEAGAN
Variety-aware GAN and Online Learning Augmented Semi-supervised Model for Knowledge Graph Entity Alignment
We will release the source code of SEAGAN later.

## Datasets
Please download the datasets [here]() and extract them into root directory.

```
ent_ids_1: entity ids in the source KG
ent_ids_2: entity ids in the target KG
ref_ent_ids: entity alignment, list of pairs like (e_s \t e_t)
triples_1: relation triples in the source KG
triples_2: relation triples in the target KG
```
## Environment

```
python==3.6.13
networkx==2.5.1
apex==0.1
pytorch==1.7.1
torch_geometric==2.0.2
cuda==11.2
```

## Running

```
CUDA_VISIBLE_DEVICES=0 python train.py --data data/DBP15K --lang EN_FR
```
