from logging import debug
import random
from turtle import distance
import pytorch_lightning as pl
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
# from transformers.utils.dummy_pt_objects import PrefixConstrainedLogitsProcessor
import networkx as nx
from .base import BaseLitModel
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from functools import partial
from .utils import rank_score, acc, LabelSmoothSoftmaxCEV1

from typing import Callable, Iterable, List

def pad_distance(pad_length, distance):
    pad = nn.ConstantPad2d(padding=(0, pad_length, 0, pad_length), value=float('-inf'))
    distance = pad(distance)
    return distance

def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

def decode(output_ids, tokenizer):
    return lmap(str.strip, tokenizer.batch_decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True))

class TransformerLitModel(BaseLitModel):
    def __init__(self, model, args, tokenizer=None, data_config={}):
        super().__init__(model, args)
        self.save_hyperparameters(args)
        if args.bce:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        elif args.label_smoothing != 0.0:
            self.loss_fn = LabelSmoothSoftmaxCEV1(lb_smooth=args.label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        self.best_acc = 0
        self.first = True
        self.entity = []
        self.relation = []

        self.symmetric_relations = {'Relation1'}
        self.anti = []
        self.inverse_relations_map = {
            'Relation1': 'Relation1_inv'
        }
        self.compose_relations_map = {
            ('Relation1', 'Relation2'): 'Relation_com'
        }

        self.tokenizer = tokenizer
        self.num_heads = 12
        self.__dict__.update(data_config)
        # resize the word embedding layer
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.decode = partial(decode, tokenizer=self.tokenizer)
        if args.pretrain:
            self._freaze_attention()
        elif "ind" in args.data_dir:
            # for inductive setting, use feeaze the word embedding
            self._freaze_word_embedding()
        
        self.graph_token_virtual_distance = nn.Embedding(1, self.num_heads)
        

    def forward(self, x):
        return self.model(x)

    """
    triples : (entity1, relation, entity2)
    G: The Graph structure of triples created by networkx.
    """
    def build_graph(self, triples):
        G = nx.Graph()
        for (entity1, relation, entity2) in triples:
            G.add_edge(entity1, entity2, relation=relation)
        return G

    ## node2vec思想实现
    def node2vec_walk(self, G, start_node, walk_length, p=1, q=1):
        walk = [start_node]
        while len(walk) < walk_length:
            current_node = walk[-1]
            neighbors = list(G.neighbors(current_node))
            if len(neighbors) == 0:
                break
            if len(walk) == 1:
                walk.append(random.choice(neighbors))
            else:
                prev_node = walk[-2]
                prob = []
                for neighbor in neighbors:
                    if neighbor == prev_node:
                        prob.append(1 / p)
                    elif G.has_edge(neighbor, prev_node):
                        prob.append(1)
                    else:
                        prob.append(1 / q)
                prob = [p / sum(prob) for p in prob]
                next_node = random.choices(neighbors, weights=prob, k=1)[0]
                walk.append(next_node)
        return walk

    # 3. 生成多个随机游走序列
    def generate_walks_with_relations(self, G, num_walks, walk_length, p=1, q=1):
        walks = []
        relations = []

        nodes = list(G.nodes())
        for _ in range(num_walks):
            random_node = random.choice(nodes)
            walk, walk_relations = self.node2vec_walk_with_relations(G, random_node, walk_length, p, q)
            walks.append(walk)
            relations.append(walk_relations)

        return walks, relations

    def node2vec_walk_with_relations(self, G, start_node, walk_length, p=1, q=1):
        walk_length = 2 * walk_length - 1
        walk = [start_node]
        walk_relations = []

        while len(walk) < walk_length:
            current_node = walk[-1]
            neighbors = list(G.neighbors(current_node))
            if len(neighbors) == 0:
                break
            if len(walk) == 1:
                chosen_node = random.choice(neighbors)
                walk_relations.append(G[current_node][chosen_node]['relation'])
                walk.append(chosen_node)
            else:
                prev_node = walk[-2]
                prob = []
                for neighbor in neighbors:
                    if neighbor == prev_node:
                        prob.append(1 / p)
                    elif G.has_edge(neighbor, prev_node):
                        prob.append(1)
                    else:
                        prob.append(1 / q)
                prob = [p / sum(prob) for p in prob]
                chosen_node = random.choices(neighbors, weights=prob, k=1)[0]
                walk_relations.append(G[current_node][chosen_node]['relation'])
                walk.append(chosen_node)

        return walk, walk_relations


    def get_anchor(self, sample):
        # 转换sample为list，便于操作
        sample_list = sample.tolist()

        # 从序列的尾部开始检查
        for i in range(len(sample_list) - 2, 0, -2):  # 遍历实体位置，从后往前
            entity = sample_list[i]
            relation = sample_list[i - 1]

            # 检查对称关系
            if relation in self.symmetric_relations:
                new_path = [sample_list[i + 1], relation, entity]
                # 从entity随机游走
                new_path.extend(self.node2vec_walk_with_relations(entity))
                return new_path

            # 检查逆关系
            if relation in self.inverse_relations_map:
                new_path = [sample_list[i + 1], self.inverse_relations_map[relation], entity]
                # 从entity随机游走
                new_path.extend(self.node2vec_walk_with_relations(entity))
                return new_path

            # 检查组合关系
            if i >= 3:  # 需要确保有足够的元素用于检查组合关系
                r1_r2 = (sample_list[i - 3], relation)
                if r1_r2 in self.compose_relations_map:
                    new_path = [sample_list[i - 4], self.compose_relations_map[r1_r2], sample_list[i + 1]]
                    # 从i + 1位置的实体随机游走
                    new_path.extend(self.node2vec_walk_with_relations(sample_list[i + 1]))
                    return new_path

        # 如果都没有匹配到，处理为反对称关系
        # 替换relation并从entity随机游走
        new_path = sample_list[:i]
        new_path.extend(self.node2vec_walk_with_relations(entity))

        return sample

    def get_negative(self, sample):
        sample_list = sample.tolist()

        # 随机选择一个位置
        random_idx = random.randint(0, len(sample_list) - 1)

        # 替换实体或关系
        if random_idx % 2 == 0:  # 如果是偶数位置，那么替换实体
            replacement = random.choice(random.choice(self.entity))
        else:  # 否则，替换关系
            replacement = random.choice(random.choice(self.relation))

        # 替换选定的位置
        sample_list[random_idx] = replacement

        # 返回替换后的负样本
        return torch.tensor(sample_list)

        return sample

    def triplet_loss(self, batch, margin=1.0):
        # 正样本
        positive = batch[0]

        # 锚样本
        anchor = self.get_anchor(positive)

        # 计算每一个负样本与锚样本的Triplet Loss
        losses = []
        for i in range(1, len(batch)):
            negative = self.get_negative(batch[i])

            # 使用欧式距离计算
            distance_positive = F.pairwise_distance(anchor, positive)
            distance_negative = F.pairwise_distance(anchor, negative)

            # 计算Triplet Loss
            loss = F.relu(distance_positive - distance_negative + margin)
            losses.append(loss)

        # 返回平均Triplet Loss
        return torch.mean(torch.stack(losses))

    ###Info
    def create_positive_pair(self, sample):
        # 使用sample创建正样本对
        positive_view1 = sample
        positive_view2 = sample[:-2] + self.node2vec_walk_with_relations(sample[-2])
        return positive_view1, positive_view2

    def info_nce_loss(self, batch, batch_idx, tau=0.5):
        # 创建正样本对
        q, k_plus = self.create_positive_pair(batch[0])
        q_feature = batch_idx(q)
        k_plus_feature = batch_idx(k_plus)

        # 计算正样本特征之间的点积
        positive_dot = (q_feature * k_plus_feature).sum(dim=-1)

        # 计算负样本特征与查询样本特征之间的点积
        negative_dots = [(q_feature * batch_idx(negative_sample)).sum(dim=-1) for negative_sample in batch[1:]]

        # 计算对比损失
        logits = torch.cat([positive_dot.view(1), torch.stack(negative_dots)])
        labels = torch.zeros(len(logits)).long().to(logits.device)
        loss = F.cross_entropy(logits / tau, labels)

        return loss

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        # if batch_idx == 0:
        #     print('\n'.join(self.decode(batch['input_ids'][:1])))

        labels = batch.pop("labels")
        label = batch.pop("label")
        
        input_ids = batch['input_ids']

        # edited by bizhen
        distance_attention = batch.pop("distance_attention")
        distance_attention = distance_attention
        
        graph_attn_bias = torch.zeros(input_ids.size(0), input_ids.size(1), input_ids.size(1)).cuda()
        graph_attn_bias[:, 1:, 1:][distance_attention == float('-inf')] = float('-inf')
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + distance_attention.unsqueeze(1)
        
        if self.args.use_global_node:
            t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
            graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
            graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t
        
        # edited by bizhen

        if self.args.add_attn_bias:
            logits = self.model(**batch, return_dict=True, distance_attention=graph_attn_bias).logits
        else:
            logits = self.model(**batch, return_dict=True, distance_attention=None).logits


        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_logits = logits[torch.arange(bs), mask_idx][:, self.entity_id_st:self.entity_id_ed]

        assert mask_idx.shape[0] == bs, "only one mask in sequence!"
        if self.args.bce:
            loss = self.loss_fn(mask_logits, labels)
        else:
            loss = self.loss_fn(mask_logits, label)

        ##info_nce_loss = self.info_nce_loss(batch, batch_idx)

        triplet_loss = self.triplet_loss(batch, batch_idx, margin=1.0)

        return loss + triplet_loss

    def _eval(self, batch, batch_idx):
        # single label
        labels = batch.pop("labels")    
        label = batch.pop('label')
        
        input_ids = batch['input_ids']
        
        # edited by bizhen
        distance_attention = batch.pop("distance_attention")
        distance_attention = distance_attention
        
        graph_attn_bias = torch.zeros(input_ids.size(0), input_ids.size(1), input_ids.size(1)).cuda()
        graph_attn_bias[:, 1:, 1:][distance_attention == float('-inf')] = float('-inf')
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + distance_attention.unsqueeze(1)
        
        if self.args.use_global_node:
            t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
            graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
            graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t
        
        if self.args.add_attn_bias:
            logits = self.model(**batch, return_dict=True, distance_attention=graph_attn_bias).logits[:, :, self.entity_id_st:self.entity_id_ed]
        else:
            logits = self.model(**batch, return_dict=True, distance_attention=None).logits[:, :, self.entity_id_st:self.entity_id_ed]
            
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bsz = input_ids.shape[0]
        logits = logits[torch.arange(bsz), mask_idx]
        # get the entity ranks
        # filter the entity
        assert labels[0][label[0]], "correct ids must in filiter!"
        labels[torch.arange(bsz), label] = 0
        assert logits.shape == labels.shape
        logits += labels * -100 # mask entityj
        # for i in range(bsz):
        #     logits[i][labels]

        _, outputs = torch.sort(logits, dim=1, descending=True)
        _, outputs = torch.sort(outputs, dim=1)
        ranks = outputs[torch.arange(bsz), label].detach().cpu() + 1
        

        return dict(ranks = np.array(ranks))

    def validation_step(self, batch, batch_idx):
        result = self._eval(batch, batch_idx)
        return result

    def validation_epoch_end(self, outputs) -> None:
        ranks = np.concatenate([_['ranks'] for _ in outputs])
        total_ranks = ranks.shape[0]

        if not self.args.pretrain:
            l_ranks = ranks[np.array(list(np.arange(0, total_ranks, 2)))]
            r_ranks = ranks[np.array(list(np.arange(0, total_ranks, 2))) + 1]
            self.log("Eval/lhits10", (l_ranks<=10).mean())
            self.log("Eval/rhits10", (r_ranks<=10).mean())

        hits20 = (ranks<=20).mean()
        hits10 = (ranks<=10).mean()
        hits3 = (ranks<=3).mean()
        hits1 = (ranks<=1).mean()

        self.log("Eval/hits10", hits10)
        self.log("Eval/hits20", hits20)
        self.log("Eval/hits3", hits3)
        self.log("Eval/hits1", hits1)
        self.log("Eval/mean_rank", ranks.mean())
        self.log("Eval/mrr", (1. / ranks).mean())
        self.log("hits10", hits10, prog_bar=True)
        self.log("hits1", hits1, prog_bar=True)
   

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        # ranks = self._eval(batch, batch_idx)
        result = self._eval(batch, batch_idx)
        # self.log("Test/ranks", np.mean(ranks))

        return result

    def test_epoch_end(self, outputs) -> None:
        ranks = np.concatenate([_['ranks'] for _ in outputs])

        hits20 = (ranks<=20).mean()
        hits10 = (ranks<=10).mean()
        hits3 = (ranks<=3).mean()
        hits1 = (ranks<=1).mean()

        self.log("Test/hits10", hits10)
        self.log("Test/hits20", hits20)
        self.log("Test/hits3", hits3)
        self.log("Test/hits1", hits1)
        self.log("Test/mean_rank", ranks.mean())
        self.log("Test/mrr", (1. / ranks).mean())

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer_class(self.parameters(), lr=self.lr, eps=1e-8)
        }
    
    def _freaze_attention(self):
        for k, v in self.model.named_parameters():
            if "word" not in k:
                v.requires_grad = False
            else:
                print(k)
    
    def _freaze_word_embedding(self):
        for k, v in self.model.named_parameters():
            if "word" in k:
                print(k)
                v.requires_grad = False

    @staticmethod
    def add_to_argparse(parser):
        parser = BaseLitModel.add_to_argparse(parser)

        parser.add_argument("--label_smoothing", type=float, default=0.1, help="")
        parser.add_argument("--bce", type=int, default=0, help="")
        return parser
