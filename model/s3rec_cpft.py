r"""
S3Rec
################################################

Reference:
    Kun Zhou and Hui Wang et al. "S^3-Rec: Self-Supervised Learning
    for Sequential Recommendation with Mutual Information Maximization"
    In CIKM 2020.

Reference code:
    https://github.com/RUCAIBox/CIKM2020-S3Rec

"""

import random
import pdb
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, MLPLayers
from recbole.model.loss import BPRLoss


class S3Rec(SequentialRecommender):
    r"""
    S3Rec is the first work to incorporate self-supervised learning in
    sequential recommendation.

    NOTE:
        Under this framework, we need reconstruct the pretraining data,
        which would affect the pre-training speed.
    """

    def __init__(self, config, dataset):
        super(S3Rec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.FEATURE_FIELD = config['item_attribute']
        self.FEATURE_LIST = self.FEATURE_FIELD + config['LIST_SUFFIX']
        self.train_stage = config['train_stage']  # pretrain or finetune
        self.pre_model_path = config['pre_model_path']  # We need this for finetune
        self.mask_ratio = config['mask_ratio']
        self.aap_weight = config['aap_weight']
        self.mip_weight = config['mip_weight']
        self.map_weight = config['map_weight']
        self.sp_weight = config['sp_weight']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # load dataset info
        self.plm_size = dataset.plm_size
        self.n_items = dataset.item_num + 1  # for mask token
        self.n_features = self.mask_token = self.n_items - 1

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
        # modules shared by pre-training stage and fine-tuning stage
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.feature_embedding = dataset.plm_embedding
        self.adapter = MLPLayers(config['adapter_layers'])

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # modules for pretrain
        # add unique dense layer for 4 losses respectively
        self.aap_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.mip_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.map_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.sp_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.loss_fct = nn.BCELoss(reduction='none')

        # modules for finetune
        if self.loss_type == 'BPR' and self.train_stage == 'finetune':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE' and self.train_stage == 'finetune':
            self.loss_fct = nn.CrossEntropyLoss()
        elif self.train_stage == 'finetune':
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        assert self.train_stage in ['pretrain', 'finetune']
        if self.train_stage == 'pretrain':
            self.apply(self._init_weights)
        else:
            # load pretrained model for finetune
            pretrained = torch.load(self.pre_model_path)
            self.logger.info(f'Load pretrained model from {self.pre_model_path}')
            self.load_state_dict(pretrained['state_dict'])

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _associated_attribute_prediction(self, sequence_output, feature_embedding):
        sequence_output = self.aap_norm(sequence_output)  # [B L H]
        sequence_output = sequence_output.view([-1, sequence_output.size(-1), 1])  # [B*L H 1]
        # [feature_num H] [B*L H 1] -> [B*L feature_num 1]
        score = torch.matmul(feature_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1))  # [B*L feature_num]

    def _masked_item_prediction(self, sequence_output, target_item_emb):
        sequence_output = self.mip_norm(sequence_output.view([-1, sequence_output.size(-1)]))  # [B*L H]
        target_item_emb = target_item_emb.view([-1, sequence_output.size(-1)])  # [B*L H]
        score = torch.mul(sequence_output, target_item_emb)  # [B*L H]
        return torch.sigmoid(torch.sum(score, -1))  # [B*L]

    def _masked_attribute_prediction(self, sequence_output, feature_embedding):
        sequence_output = self.map_norm(sequence_output)  # [B L H]
        sequence_output = sequence_output.view([-1, sequence_output.size(-1), 1])  # [B*L H 1]
        # [feature_num H] [B*L H 1] -> [B*L feature_num 1]
        score = torch.matmul(feature_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1))  # [B*L feature_num]

    def _segment_prediction(self, context, segment_emb):
        context = self.sp_norm(context)
        score = torch.mul(context, segment_emb)  # [B H]
        return torch.sigmoid(torch.sum(score, dim=-1))  # [B]

    def get_attention_mask(self, sequence, bidirectional=True):
        """
        In the pre-training stage, we generate bidirectional attention mask for multi-head attention.

        In the fine-tuning stage, we generate left-to-right uni-directional attention mask for multi-head attention.
        """
        attention_mask = (sequence > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        if not bidirectional:
            max_len = attention_mask.size(-1)
            attn_shape = (1, max_len, max_len)
            subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
            subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
            subsequent_mask = subsequent_mask.long().to(sequence.device)
            extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, bidirectional=True):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        attention_mask = self.get_attention_mask(item_seq, bidirectional=bidirectional)
        trm_output = self.trm_encoder(input_emb, attention_mask, output_all_encoded_layers=True)
        seq_output = trm_output[-1]  # [B L H]
        return seq_output

    def pretrain(
        self, features, masked_item_sequence, pos_items, neg_items, masked_segment_sequence, pos_segment, neg_segment
    ):
        """Pretrain out model using four pre-training tasks:

            1. Associated Attribute Prediction

            2. Masked Item Prediction

            3. Masked Attribute Prediction

            4. Segment Prediction
        """
        # Encode masked sequence
        sequence_output = self.forward(masked_item_sequence)

        feature_embedding = self.adapter(self.feature_embedding.weight)
        # AAP
        aap_score = self._associated_attribute_prediction(sequence_output, feature_embedding)
        aap_loss = self.loss_fct(aap_score, features.view(-1, self.n_features).float())
        # only compute loss at non-masked position
        aap_mask = (masked_item_sequence != self.mask_token).float() * \
                   (masked_item_sequence != 0).float()
        aap_loss = torch.sum(aap_loss * aap_mask.flatten().unsqueeze(-1))

        # MIP
        pos_item_embs = self.item_embedding(pos_items)
        neg_item_embs = self.item_embedding(neg_items)
        pos_score = self._masked_item_prediction(sequence_output, pos_item_embs)
        neg_score = self._masked_item_prediction(sequence_output, neg_item_embs)
        mip_distance = torch.sigmoid(pos_score - neg_score)
        mip_loss = self.loss_fct(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32))
        mip_mask = (masked_item_sequence == self.mask_token).float()
        mip_loss = torch.sum(mip_loss * mip_mask.flatten())

        # MAP
        map_score = self._masked_attribute_prediction(sequence_output, feature_embedding)
        map_loss = self.loss_fct(map_score, features.view(-1, self.n_features).float())
        map_mask = (masked_item_sequence == self.mask_token).float()
        map_loss = torch.sum(map_loss * map_mask.flatten().unsqueeze(-1))

        # SP
        # segment context
        # take the last position hidden as the context
        segment_context = self.forward(masked_segment_sequence)[:, -1, :]  # [B H]
        pos_segment_emb = self.forward(pos_segment)[:, -1, :]
        neg_segment_emb = self.forward(neg_segment)[:, -1, :]  # [B H]
        pos_segment_score = self._segment_prediction(segment_context, pos_segment_emb)
        neg_segment_score = self._segment_prediction(segment_context, neg_segment_emb)
        sp_distance = torch.sigmoid(pos_segment_score - neg_segment_score)
        sp_loss = torch.sum(self.loss_fct(sp_distance, torch.ones_like(sp_distance, dtype=torch.float32)))

        pretrain_loss = self.aap_weight * aap_loss \
                        + self.mip_weight * mip_loss \
                        + self.map_weight * map_loss \
                        + self.sp_weight * sp_loss

        return pretrain_loss

    def _neg_sample(self, item_set):  # [ , ]
        item = random.randint(1, self.n_items - 1)
        while item in item_set:
            item = random.randint(1, self.n_items - 1)
        return item

    def _padding_zero_at_left(self, sequence):
        # had truncated according to the max_length
        pad_len = self.max_seq_length - len(sequence)
        sequence = [0] * pad_len + sequence
        return sequence

    def reconstruct_pretrain_data(self, item_seq, item_seq_len):
        """Generate pre-training data for the pre-training stage."""
        device = item_seq.device
        batch_size = item_seq.size(0)

        associated_features = torch.zeros([batch_size, item_seq.size(1), self.n_features], dtype=torch.long, device=device)
        associated_features = associated_features.scatter(dim=2, index=item_seq.unsqueeze(-1), src=torch.ones([batch_size, item_seq.size(1), 1], dtype=torch.long, device=device))

        end_index = item_seq_len.cpu().numpy().tolist()
        item_seq = item_seq.cpu().numpy().tolist()

        # we will padding zeros at the left side
        # these will be train_instances, after will be reshaped to batch
        sequence_instances = []
        long_sequence = []
        for i, end_i in enumerate(end_index):
            sequence_instances.append(item_seq[i][:end_i])
            long_sequence.extend(item_seq[i][:end_i])

        # Masked Item Prediction and Masked Attribute Prediction
        # [B * Len]
        masked_item_sequence = []
        pos_items = []
        neg_items = []
        for instance in sequence_instances:
            masked_sequence = instance.copy()
            pos_item = instance.copy()
            neg_item = instance.copy()
            for index_id, item in enumerate(instance):
                prob = random.random()
                if prob < self.mask_ratio:
                    masked_sequence[index_id] = self.mask_token
                    neg_item[index_id] = self._neg_sample(instance)
            masked_item_sequence.append(self._padding_zero_at_left(masked_sequence))
            pos_items.append(self._padding_zero_at_left(pos_item))
            neg_items.append(self._padding_zero_at_left(neg_item))

        # Segment Prediction
        masked_segment_list = []
        pos_segment_list = []
        neg_segment_list = []
        for instance in sequence_instances:
            if len(instance) < 2:
                masked_segment = instance.copy()
                pos_segment = instance.copy()
                neg_segment = instance.copy()
            else:
                sample_length = random.randint(1, len(instance) // 2)
                start_id = random.randint(0, len(instance) - sample_length)
                neg_start_id = random.randint(0, len(long_sequence) - sample_length)
                pos_segment = instance[start_id:start_id + sample_length]
                neg_segment = long_sequence[neg_start_id:neg_start_id + sample_length]
                masked_segment = instance[:start_id] + [self.mask_token] * sample_length \
                                 + instance[start_id + sample_length:]
                pos_segment = [self.mask_token] * start_id + pos_segment + \
                              [self.mask_token] * (len(instance) - (start_id + sample_length))
                neg_segment = [self.mask_token] * start_id + neg_segment + \
                              [self.mask_token] * (len(instance) - (start_id + sample_length))
            masked_segment_list.append(self._padding_zero_at_left(masked_segment))
            pos_segment_list.append(self._padding_zero_at_left(pos_segment))
            neg_segment_list.append(self._padding_zero_at_left(neg_segment))

        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(batch_size, -1)
        neg_items = torch.tensor(neg_items, dtype=torch.long, device=device).view(batch_size, -1)
        masked_segment_list = torch.tensor(masked_segment_list, dtype=torch.long, device=device).view(batch_size, -1)
        pos_segment_list = torch.tensor(pos_segment_list, dtype=torch.long, device=device).view(batch_size, -1)
        neg_segment_list = torch.tensor(neg_segment_list, dtype=torch.long, device=device).view(batch_size, -1)

        return associated_features, masked_item_sequence, pos_items, neg_items, \
               masked_segment_list, pos_segment_list, neg_segment_list

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        # calibration loss
        test_item_emb = self.item_embedding.weight
        cor_cal_seq_output = self.forward(item_seq, bidirectional=False)
        cor_cal_logits = torch.matmul(cor_cal_seq_output, test_item_emb.transpose(0, 1))

        train_seq_len = item_seq_len - 1
        # Replace values at the specified indices with 0
        mask = torch.zeros_like(item_seq, dtype=torch.bool)
        mask[torch.arange(item_seq.size(0)), train_seq_len] = True
        train_item_seq = item_seq.masked_fill(mask, 0)
        
        # pretrain
        if self.train_stage == 'pretrain':
            features, masked_item_sequence, pos_items, neg_items, \
            masked_segment_sequence, pos_segment, neg_segment \
                = self.reconstruct_pretrain_data(train_item_seq, train_seq_len)

            loss = self.pretrain(
                features, masked_item_sequence, pos_items, neg_items, masked_segment_sequence, pos_segment, neg_segment
            )
        # finetune
        else:
            # Extract the values at item_seq_len - 1 index for each row
            pos_items = item_seq[torch.arange(item_seq.size(0)), train_seq_len]
            # we use uni-directional attention in the fine-tuning stage
            seq_output = self.forward(train_item_seq, bidirectional=False)
            seq_output = self.gather_indexes(seq_output, item_seq_len - 1)

            if self.loss_type == 'BPR':
                neg_items = interaction[self.NEG_ITEM_ID]
                pos_items_emb = self.item_embedding(pos_items)
                neg_items_emb = self.item_embedding(neg_items)
                pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
                neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
                total_loss = self.loss_fct(pos_score, neg_score)
            else:  # self.loss_type = 'CE'
                test_item_emb = self.item_embedding.weight
                logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
                loss = torch.tensor(0.0)
                if self.calculate_prediction_loss:
                    loss = self.loss_fct(logits, pos_items)
                    if torch.isnan(loss):
                        raise ValueError("Training prediction loss is nan")
        
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
        mask[index_tuple[0], index_tuple[1]] = False
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

        return dis_loss
    
    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, bidirectional=False)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, bidirectional=False)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_items_emb = self.item_embedding.weight[:self.n_items - 1]  # delete masked token
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
