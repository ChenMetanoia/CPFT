# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
import pdb
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # CPFT specific parameters
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.gamma = config['gamma']
        self.dis_k = config['dis_k']
        self.calculate_calibration_loss = config['calculate_calibration_loss']
        self.calculate_distance_loss = config['calculate_distance_loss']
        self.calculate_prediction_loss = config['calculate_prediction_loss']
        self.calculate_by = config['calculate_by']
        self.distance_type = config['distance_type']
        
        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        # calibration loss
        test_item_emb = self.item_embedding.weight
        cor_cal_seq_output = self.forward(item_seq, item_seq_len)
        cor_cal_logits = torch.matmul(cor_cal_seq_output, test_item_emb.transpose(0, 1))

        train_seq_len = item_seq_len - 1
        
        # Replace values at the specified indices with 0
        mask = torch.zeros_like(item_seq, dtype=torch.bool)
        mask[torch.arange(item_seq.size(0)), train_seq_len] = True
        train_item_seq = item_seq.masked_fill(mask, 0)
        
        train_seq_output = self.forward(train_item_seq, train_seq_len)
        
        # Extract the values at item_seq_len - 1 index for each row
        pos_items = item_seq[torch.arange(item_seq.size(0)), train_seq_len]
        
        logits = torch.matmul(train_seq_output, test_item_emb.transpose(0, 1))
        
        if self.loss_type == "CE":
            # self.loss_type = 'CE'
            loss = torch.tensor(0.0)
            if self.calculate_prediction_loss:
                loss = self.loss_fct(logits, pos_items)
                if torch.isnan(loss):
                    raise ValueError("Training prediction loss is nan")
        elif self.loss_type == "BPR":
            def replace_equal_elements(pos_items, neg_items):
                """
                Replace elements in neg_items that are equal to elements in pos_items.
                
                :param pos_items: torch.Tensor, positive items.
                :param neg_items: torch.Tensor, negative items to be checked against pos_items.
                """
                # Create a mask for elements that are the same in pos_items and neg_items
                mask = pos_items == neg_items

                # Filter neg_item_pool to exclude any items present in pos_items
                unique_neg_items = torch.unique(neg_items)

                # Replace elements in neg_items where mask is True
                while mask.any():
                    neg_items[mask] = unique_neg_items[torch.randint(0, len(unique_neg_items), (mask.sum(),))]
                    mask = pos_items == neg_items

                return neg_items
            
            neg_items = interaction[self.NEG_ITEM_ID]
            neg_items = replace_equal_elements(pos_items, neg_items)
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(train_seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(train_seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        
        # calculating the plug-in loss
        comparison_result, prob = self._calculate_comparsion_result(logits, pos_items, cor_cal_logits)
        
       # calibration loss
        calib_loss = torch.tensor(0.0)
        if self.calculate_calibration_loss:
            calib_loss = self.calculate_set_size_loss(comparison_result, prob)
            if torch.isnan(calib_loss):
                raise ValueError("Training calibration loss is nan")
        # distance loss
        dist_loss = torch.tensor(0.0)
        if self.calculate_distance_loss:
            dist_loss = self.calculate_dist_loss(test_item_emb, comparison_result, pos_items)
            if torch.isnan(dist_loss):
                raise ValueError("Training distance loss is nan")
        total_loss = loss + self.beta * calib_loss + self.gamma * dist_loss
        return total_loss

    def calculate_eta(self, logits, pos_items):
        '''Calculate the probability of the target item
        logits: [B n_items]
        pos_items: [B]
        return: [B]
        '''  
        probs = nn.functional.softmax(logits, dim=1)
        cal_scores = probs[torch.arange(logits.size(0)), pos_items]
        # calculate the cumulative sum of the probability
        target_probability = torch.quantile(cal_scores, self.alpha * (1 + 1/ logits.size(0)), interpolation='higher')
        return target_probability
    
    def _calculate_comparsion_result(self, logits, cor_cal_pos_items, cor_cal_logits):
        # get the treshold from the training (fine tuning) data
        threshold = self.calculate_eta(logits, cor_cal_pos_items)
        prob = nn.functional.softmax(cor_cal_logits, dim=1)
        comparison_result = torch.relu(prob - threshold)
        return comparison_result, prob
    
    def calculate_set_size_loss(self, comparison_result, prob):
        '''
        calculate the set size loss on the calibration set
        no ground truth label needed
        '''
        output_tensor = torch.where(comparison_result > 0, torch.ones_like(prob), prob)
        C_score = torch.sum(output_tensor, dim=1)
        return torch.mean(C_score, dtype=torch.float)
    
    def calculate_dist_loss(self, test_item_emb, comparison_result, pos_items):
        calculate_by = self.calculate_by
        distance_type = self.distance_type
        
        # Check calculate_by and distance_type early
        if calculate_by not in ['cosine', 'euclidean']:
            raise ValueError('Unknown calculate_by')
        if distance_type not in ['min', 'max', 'avg']:
            raise ValueError('Unknown distance_type')

        pos_item_emb = test_item_emb[pos_items]
        index_tuple = torch.where(comparison_result > 0)
        
        # Compute similarity or distance
        if calculate_by == 'cosine':
            similarity_matrix = torch.matmul(pos_item_emb, test_item_emb.t())  # Transpose for cosine
            if distance_type == 'min':
                mask_value = -float('inf')
            elif distance_type == 'max':
                mask_value = float('inf')
            else:
                mask_value = float('nan')
        else:  # calculate_by == 'euclidean'
            similarity_matrix = torch.cdist(pos_item_emb, test_item_emb, p=2)
            if distance_type == 'min':
                mask_value = float('inf')
            elif distance_type == 'max':
                mask_value = -float('inf')
            else:
                mask_value = float('nan')
                
        # Apply mask
        mask = torch.ones_like(similarity_matrix, dtype=bool)
        mask[index_tuple] = False
        # generate tensor of index starting from 0 to the shape of pos_items
        row_idx = torch.arange(pos_items.size(0))
        mask[row_idx, pos_items] = True
        similarity_matrix[mask] = mask_value
        # Compute the distance loss
        if distance_type == 'min' or distance_type == 'max':
            if calculate_by == 'cosine':
                largest = distance_type == 'min'
                top_k_values, _ = similarity_matrix.topk(self.dis_k, dim=1, largest=largest)
                top_k_values = torch.where(top_k_values == mask_value, float('nan'), top_k_values)
                dis_loss = 1 - torch.nanmean(top_k_values)
            elif calculate_by == 'euclidean':
                largest = distance_type == 'max'
                top_k_values, _ = similarity_matrix.topk(self.dis_k, dim=1, largest=largest)
                top_k_values = torch.where(top_k_values == mask_value, float('nan'), top_k_values)
                dis_loss = torch.nanmean(top_k_values)
            else:
                raise ValueError('Unknown calculate_by')
        else:
            dis_loss = torch.nanmean(similarity_matrix)
            
        if torch.isinf(dis_loss) or torch.isnan(dis_loss):
            pdb.set_trace()

        clipped_dis_loss = torch.max(dis_loss, torch.tensor(0.0))
        return clipped_dis_loss
    
    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
