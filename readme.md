# TracKGE
Code for paper:"TracKGE: Transformer with Relation-pattern Adaptive Contrastive Learning for Knowledge Graph Embedding".

**We have uploaded some of the core code. We will complete the other codes,  training and validation scripts after the article is accepted.** 

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
