import copy
import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)


class UniSRec(SASRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.train_stage = config['train_stage']
        self.temperature = config['temperature']
        self.lam = config['lambda']
        self.alpha = config['alpha']
        
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
        # self.subset_predict = config.get('subset_predict', False)

        assert self.train_stage in [
            'pretrain', 'inductive_ft', 'transductive_ft'
        ], f'Unknown train stage: [{self.train_stage}]'

        if self.train_stage in ['pretrain', 'inductive_ft']:
            self.item_embedding = None
            # for `transductive_ft`, `item_embedding` is defined in SASRec base model
        if self.train_stage in ['inductive_ft', 'transductive_ft']:
            # `plm_embedding` in pre-train stage will be carried via dataloader
            self.plm_embedding = copy.deepcopy(dataset.plm_embedding)

        self.moe_adaptor = MoEAdaptorLayer(
            config['n_exps'],
            config['adaptor_layers'],
            config['adaptor_dropout_prob']
        )

    def forward(self, item_seq, item_emb, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        if self.train_stage == 'transductive_ft':
            input_emb = input_emb + self.item_embedding(item_seq)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def seq_item_contrastive_task(self, seq_output, same_pos_id, interaction):
        pos_items_emb = self.moe_adaptor(interaction['pos_item_emb'])
        pos_items_emb = F.normalize(pos_items_emb, dim=1)

        pos_logits = (seq_output * pos_items_emb).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def seq_seq_contrastive_task(self, seq_output, same_pos_id, interaction):
        item_seq_aug = interaction[self.ITEM_SEQ + '_aug']
        item_seq_len_aug = interaction[self.ITEM_SEQ_LEN + '_aug']
        item_emb_list_aug = self.moe_adaptor(interaction['item_emb_list_aug'])
        seq_output_aug = self.forward(item_seq_aug, item_emb_list_aug, item_seq_len_aug)
        seq_output_aug = F.normalize(seq_output_aug, dim=1)

        pos_logits = (seq_output * seq_output_aug).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, seq_output_aug.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()
    
    def pretrain(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(interaction['item_emb_list'])
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        seq_output = F.normalize(seq_output, dim=1)

        # Remove sequences with the same next item
        pos_id = interaction['item_id']
        same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0))
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device))

        loss_seq_item = self.seq_item_contrastive_task(seq_output, same_pos_id, interaction)
        loss_seq_seq = self.seq_seq_contrastive_task(seq_output, same_pos_id, interaction)
        loss = loss_seq_item + self.lam * loss_seq_seq
        return loss

    def calculate_eta(self, logits, pos_items):
        '''Calculate the probability of the target item
        logits: [B n_items]
        pos_items: [B]
        return: [B]
        '''
        # convert logits to probability
        probs = nn.functional.softmax(logits, dim=1)
        cal_scores = probs[torch.arange(logits.size(0)), pos_items]
        # compute α-percentile of positive-item probabilities
        target_probability = torch.quantile(cal_scores, self.alpha * (1 + 1/ logits.size(0)), interpolation='higher')
        return target_probability
    
    def _calucate_rank_based_comparsion_result(self, train_logits, train_pos_items, val_logits, lmd=0.03):
        # get the treshold from the training (fine tuning) data
        threshold = self.calculate_eta(train_logits, train_pos_items)
        # convert logits to probability
        prob = nn.functional.softmax(val_logits, dim=1).detach().cpu()
        sort_idx = torch.argsort(prob, dim=1, descending=True)
        mask = torch.eq(sort_idx, train_pos_items.view(-1, 1).detach().cpu())
        positions = mask.nonzero()[:, 1]
        batch_scores = torch.zeros_like(positions, dtype=torch.float)
        # if the ranking is at the top
        top_mask = torch.where(positions==0)[0]
        top_prob = prob[top_mask,:]
        top_prob_idx = sort_idx[top_mask,:][:,0]
        top_res = top_prob[torch.arange(top_prob.size(0)), top_prob_idx]
        # u \cdot \hat(\pi)_{max}(x)
        batch_scores[top_mask] = top_res * torch.rand(top_res.size(0))
        # if the ranking is not at the top
        non_top_mask = torch.where(positions!=0)[0]
        non_top_positions = positions[non_top_mask]
        non_top_res = prob[non_top_mask, non_top_positions]
        batch_scores[non_top_mask] = non_top_res + (non_top_positions - 2 + torch.rand(non_top_positions.size(0))) * lmd
        # \hat(\pi)_{max}(x) + (rank(x)-2+u)\cdot \alpha
        comparison_result = torch.relu(prob - (1 - threshold))
        return comparison_result, prob
    
    def _calculate_comparsion_result(self, logits, cor_cal_pos_items, cor_cal_logits):
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
    
    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            return self.pretrain(interaction)
        # Loss for fine-tuning
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        # calibration loss
        cor_cal_item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        cor_cal_seq_output = self.forward(item_seq, cor_cal_item_emb_list, item_seq_len)
        cor_cal_seq_output = F.normalize(cor_cal_seq_output, dim=1)
        test_item_emb = self.moe_adaptor(self.plm_embedding.weight)
        if self.train_stage == 'transductive_ft':
            test_item_emb = test_item_emb + self.item_embedding.weight
        test_item_emb = F.normalize(test_item_emb, dim=1)
        cor_cal_logits = torch.matmul(cor_cal_seq_output, test_item_emb.transpose(0, 1)) / self.temperature

        train_seq_len = item_seq_len - 1

        # # Replace values at the specified indices with 0
        mask = torch.zeros_like(item_seq, dtype=torch.bool)
        mask[torch.arange(item_seq.size(0)), train_seq_len] = True
        train_item_seq = item_seq.masked_fill(mask, 0)

        # Extract the values at item_seq_len - 1 index for each row
        pos_items = item_seq[torch.arange(item_seq.size(0)), train_seq_len]
        
        train_item_emb_list = self.moe_adaptor(self.plm_embedding(train_item_seq))
        train_seq_output = self.forward(train_item_seq, train_item_emb_list, train_seq_len)
        train_seq_output = F.normalize(train_seq_output, dim=1)
        logits = torch.matmul(train_seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        
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

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_items_emb = self.moe_adaptor(self.plm_embedding.weight)
        if self.train_stage == 'transductive_ft':
            test_items_emb = test_items_emb + self.item_embedding.weight

        seq_output = F.normalize(seq_output, dim=-1)
        test_items_emb = F.normalize(test_items_emb, dim=-1)

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
