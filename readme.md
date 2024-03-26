# TracKGE
Code for paper:"TracKGE: Transformer with Relation-pattern Adaptive Contrastive Learning for Knowledge Graph Embedding".

We have uploaded some of the core code and test samples for verification, and other code will be gradually open source in the future.


# Environments
- python(3.8.10)
- Pytorch (1.7.1)
- Nvidia Tesla A100 GPU

# Requirements
To run the codes, the requirements is needed:
```shell
pip install -r requirements.txt
```

The structure of files is:

```
 ── TracKGE
    ├── dataset
    │   ├── fb15k-237
    │   ├── wn18rr
    │   ├── fb15k
    │   ├── wn18
    ├── pytorch_lightning_model
    │   ├── _init_.py
    │   ├── base.py
    │   ├── transformer.py
    │   └── utils.py
    ├── scripts
    │   ├── fb15k-237
    │   ├── fb15k
    │   ├── wn18
    │   └── wn18rr
    ├── main.py
    └── requirements.txt

```


# Verificate the model.
## Get checkpoint.
1. Download the checkpoint file of TracKGE.(Recommend)

    - https://drive.google.com/drive/folders/1qbG-ZtDLZRsJPjr04HTNbCyej1E_qSPI?usp=drive_link

    - https://pan.baidu.com/s/1WYS1LXknFJjbIbwbI-0wZg

        code: new4

2. Training through the optimal combination of super parameters.

    |        |  Batch size   | lr  | Dim.|p|q|
    |  ----  | ----  |----|----|----|----|
    | FB15K  | 1024 |10^(-4)|512|1.0|0.6|
    | FB15K-237  | 512 |5×10^(-4)|1024|0.8|0.4|
    | WN18  | 512 |5×10^(-4)|256|0.6|0.6|
    | WN18RR  | 512 |10^(-4)|512|0.8|0.6|

## Script.
Enter the checkpoint path into the script.
```shell
cd TracKGE
sh scripts/fb15k-237/fb15k-237.sh
```
