import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        self.pe = nn.Embedding(max_len, dim)
        nn.init.xavier_normal_(self.pe.weight.data)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

class infoNCE_loss(nn.Module):
    def __init__(self, temp_init, hdim):
        super().__init__()
        self.temp = nn.Parameter(torch.ones([]) * temp_init)

        self.weight_matrix = nn.Parameter(torch.randn((hdim, hdim)))
        nn.init.xavier_normal_(self.weight_matrix)

        self.tanh = nn.Tanh()

    def calculate_loss(self, query, item, neg_item):
        positive_logit = torch.sum((query @ self.weight_matrix) * item, dim=1, keepdim=True)
        negative_logits = (query @ self.weight_matrix) @ neg_item.transpose(-2, -1)

        positive_logit, negative_logits = self.tanh(positive_logit), self.tanh(negative_logits)

        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)

        return F.cross_entropy(logits / self.temp, labels, reduction='mean')

    def forward(self, query, click_item, neg_item, neg_query):
        query_loss = self.calculate_loss(query, click_item, neg_item)
        item_loss = self.calculate_loss(click_item, query, neg_query)

        return 0.5 * (query_loss + item_loss)
    
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_unit, batch_norm=False, activation='relu', sigmoid=False, dropout=None):
        super().__init__()
        assert len(hidden_unit) >= 1
        self.sigmoid = sigmoid

        layers = []
        layers.append(nn.Linear(input_size, hidden_unit[0]))

        for i, h in enumerate(hidden_unit[:-1]):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_unit[i]))

            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activation.lower() == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            else:
                raise NotImplementedError

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(hidden_unit[i], hidden_unit[i + 1]))

        self.fc = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x)
    
class FullyConnectedLayer(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True, batch_norm=False, activation='leakyrelu'):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(embed_dim))
            if activation == 'leakyrelu':
                layers.append(torch.nn.LeakyReLU())
            elif activation == 'swish':
                layers.append(torch.nn.SiLU())
            else:
                raise ValueError('No supported activation function!')
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    
class InputFeatureGate(nn.Module):
    def __init__(self, d_model, reduction=4):
        super().__init__()
        self.d_model = d_model
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // reduction),
            nn.LeakyReLU(),
            nn.Linear(d_model // reduction, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        weight = self.gate(x)
        return x * weight

class TargetAttentiveExpert(nn.Module):
    def __init__(self, d_model, d_out, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.mlp = FullyConnectedLayer(d_model, [d_out], dropout, output_layer=False, batch_norm=False, activation='swish')

    def forward(self, expert_opinion):

        return self.mlp(expert_opinion)

class TargetAttentive_PLE_layer(nn.Module):
    def __init__(self, d_model, bottom_mlp_dims, tower_mlp_dims, task_num, shared_expert_num, specific_expert_num, dropout):
        super().__init__()
        self.d_model = d_model
        self.task_num = task_num
        self.shared_expert_num = shared_expert_num
        self.specific_expert_num = specific_expert_num
        self.layers_num = len(bottom_mlp_dims)

        self.task_experts = [[0] * self.task_num for _ in range(self.layers_num)]
        self.task_gates = [[0] * self.task_num for _ in range(self.layers_num)]
        self.share_experts = [0] * self.layers_num
        self.share_gates = [0] * self.layers_num

        self.task_self_gates = [[0] * self.task_num for _ in range(self.layers_num)]

        self.task_input_gates = [[0] * self.task_num for _ in range(self.layers_num)]
        self.share_input_gates = [0] * self.layers_num

        for i in range(self.layers_num):
            expert_d_in = self.d_model if i == 0 else bottom_mlp_dims[i - 1]
            expert_d_out = bottom_mlp_dims[i]

            self.share_input_gates[i] = InputFeatureGate(expert_d_in)
            gate_input_dim = expert_d_in

            self.share_experts[i] = nn.ModuleList([TargetAttentiveExpert(expert_d_in, expert_d_out, dropout) for _ in range(self.shared_expert_num)])
            self.share_gates[i] = nn.Sequential(nn.Linear(gate_input_dim, shared_expert_num + task_num * specific_expert_num), nn.Softmax(dim=-1))

            for j in range(task_num):
                if self.specific_expert_num == 1:
                    self.task_self_gates[i][j] = nn.Sequential(nn.Linear(gate_input_dim, 1), nn.Sigmoid())
                else:
                    self.task_self_gates[i][j] = nn.Sequential(nn.Linear(gate_input_dim, self.specific_expert_num), nn.Softmax(dim=-1))

                self.task_experts[i][j] = nn.ModuleList([TargetAttentiveExpert(expert_d_in, expert_d_out, dropout) for _ in range(self.specific_expert_num)])

                self.task_gates[i][j] = nn.Sequential(nn.Linear(gate_input_dim, shared_expert_num + specific_expert_num), nn.Softmax(dim=-1))

                self.task_input_gates[i][j] = InputFeatureGate(expert_d_in)

            self.task_experts[i] = nn.ModuleList(self.task_experts[i])
            self.task_gates[i] = nn.ModuleList(self.task_gates[i])

            self.task_self_gates[i] = nn.ModuleList(self.task_self_gates[i])
            self.task_input_gates[i] = nn.ModuleList(self.task_input_gates[i])
        
        self.task_experts = nn.ModuleList(self.task_experts)
        self.task_gates = nn.ModuleList(self.task_gates)
        self.share_experts = nn.ModuleList(self.share_experts)
        self.share_gates = nn.ModuleList(self.share_gates)

        self.task_self_gates = nn.ModuleList(self.task_self_gates)
        self.share_input_gates = nn.ModuleList(self.share_input_gates)
        self.task_input_gates = nn.ModuleList(self.task_input_gates)

        self.tower = nn.ModuleList(
            [FullyConnectedLayer(bottom_mlp_dims[-1], tower_mlp_dims, dropout, output_layer=False, batch_norm=False, activation='swish') 
             for _ in range(task_num)]
        )

        self.share_tmp = nn.Parameter(torch.tensor(1.0))
        self.task_tmp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_seq, target_emb, key_padding_mask=None):
        task_feat = [target_emb for _ in range(self.task_num + 1)]

        for i in range(self.layers_num):
            share_gated_x_seq = self.share_input_gates[i](x_seq)
            share_scores = target_emb @ share_gated_x_seq.transpose(2, 1) / (self.share_tmp) # [B, N, D] @ [B, D, L] -> [B, N, L]
            if key_padding_mask is not None:
                share_scores = share_scores.masked_fill(key_padding_mask.unsqueeze(1) == 0, float('-inf'))
            share_attn_weights = torch.softmax(share_scores, dim=-1) # [B, N, L]
            share_expert_opinion = share_attn_weights @ share_gated_x_seq
            share_output = [expert(share_expert_opinion).unsqueeze(1) for expert in self.share_experts[i]] # * [B, 1, N, D]
            
            task_output_list = []

            for j in range(self.task_num):
                task_output = []
                task_gated_x_seq = self.task_input_gates[i][j](x_seq)
                task_scores = target_emb @ task_gated_x_seq.transpose(2, 1) / (self.task_tmp) # [B, N, D] @ [B, D, L] -> [B, N, L]
                if key_padding_mask is not None:
                    task_scores = task_scores.masked_fill(key_padding_mask.unsqueeze(1) == 0, float('-inf'))
                task_attn_weights = torch.softmax(task_scores, dim=-1) # [B, N, L]
                task_expert_opinion = task_attn_weights @ task_gated_x_seq
                for k, expert in enumerate(self.task_experts[i][j]):
                    out = expert(task_expert_opinion).unsqueeze(1)
                    task_output.append(out)

                task_output_list.extend(task_output)
                mix_output = torch.cat(task_output + share_output, dim=1) # * [B, X, N, D]
                gate_input = task_feat[j]
                gate_value = self.task_gates[i][j](gate_input).unsqueeze(1)
                standard_out = torch.matmul(gate_value.transpose(2, 1), mix_output.transpose(2, 1)).squeeze(2) # [B, N, 1, X] @ [B, N, X, D] -> [B, N, D]
                
                task_feat[j] = standard_out

            if i != self.layers_num - 1:
                gate_value = self.share_gates[i](task_feat[-1]).unsqueeze(1)
                mix_output = torch.cat(task_output_list + share_output, dim=1) # * [B, X, N, D]
                task_feat[-1] = torch.matmul(gate_value.transpose(2, 1), mix_output.transpose(2, 1)).squeeze(2) # [B, N, 1, X] @ [B, N, X, D] -> [B, N, D]

        results = [self.tower[i](task_feat[i]) for i in range(self.task_num)] # * [B, N, D]
        return results

class Transformer_layer(nn.Module):
    def __init__(self, emb_size, num_heads, num_layers, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads, dim_feedforward=4*emb_size, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
    
    def forward(self, his_emb, src_key_padding_mask, src_mask=None):
        src_key_padding_mask = (src_key_padding_mask == 0)

        if src_mask is not None:
            src_mask_expand = src_mask.unsqueeze(1).expand((-1, self.num_heads, -1, -1)).reshape((-1, his_emb.size(1), his_emb.size(1)))
            his_encoded = self.transformer_encoder(src=his_emb, src_key_padding_mask=src_key_padding_mask, mask=src_mask_expand)
        else:
            his_encoded = self.transformer_encoder(src=his_emb, src_key_padding_mask=src_key_padding_mask)

        return his_encoded * (~src_key_padding_mask).unsqueeze(-1).float()
    
class BehaviorAareMHA(nn.Module):
    def __init__(self, d_model, num_behaviors, head_splits, attn_dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_behaviors = num_behaviors
        self.num_intra_heads = num_behaviors

        self.num_general_heads = head_splits.get('general', 0)
        self.num_inter_heads = head_splits.get('inter', 0)
        self.num_heads = self.num_intra_heads + self.num_general_heads + self.num_inter_heads
        self.d_head = d_model // self.num_heads

        idx = 0
        self.general_head_indices = list(range(idx, idx + self.num_general_heads))
        idx += self.num_general_heads

        self.inter_head_indices = list(range(idx, idx + self.num_inter_heads))
        idx += self.num_inter_heads

        self.intra_head_indices = list(range(idx, idx + self.num_intra_heads))
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.gate_weight = nn.Parameter(torch.empty(self.num_heads, self.d_head, self.d_head))
        self.gate_bias = nn.Parameter(torch.zeros(self.num_heads, self.d_head))
        nn.init.xavier_normal_(self.gate_weight)

        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, behavior_mask, key_padding_mask=None):
        B, L, D = x.shape

        # Q, K, V shape: (B, H, L, d_h)
        Q = self.W_q(x).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head) # (B, H, L, d_h) @ (B, H, d_h, L) = (B, H, L, L)

        q_mask_bool = behavior_mask.bool().unsqueeze(2) # (B, L, K) -> (B, L, 1, K)
        k_mask_bool = behavior_mask.bool().unsqueeze(1) # (B, L, K) -> (B, 1, L, K)

        behavior_mask_final = torch.full_like(scores, float('-inf')) # (B, H, L, L)
        behavior_mask_final[:, self.general_head_indices, :, :] = 0.0

        q_has_any_behavior = q_mask_bool.any(dim=-1) # (B, L, 1)
        k_has_any_behavior = k_mask_bool.any(dim=-1) # (B, 1, L)
        both_has_any_behavior = q_has_any_behavior & k_has_any_behavior # (B, L, L)

        shared_behaviors = q_mask_bool & k_mask_bool # (B, L, L, K)
        intra_mask_template = shared_behaviors.any(dim=-1) # (B, L, L)

        inter_mask_template = (~intra_mask_template) & both_has_any_behavior # (B, L, L)

        inter_diagonal_elements = inter_mask_template.diagonal(offset=0, dim1=-2, dim2=-1) # (B, L)
        inter_diagonal_elements.fill_(1.0)

        behavior_mask_final[:, self.inter_head_indices] = behavior_mask_final[:, self.inter_head_indices].masked_fill(inter_mask_template.unsqueeze(1), 0.0)
        
        for i, head_index in enumerate(self.intra_head_indices):
            q_is_b = q_mask_bool[:, :, :, i] # (B, L, 1)
            k_is_b = k_mask_bool[:, :, :, i] # (B, 1, L)
            intra_b_mask = q_is_b & k_is_b # (B, L, L)

            # NOTE:intra_b_mask's diagonal elements can set to 1.0
            intra_diagonal_elements = intra_b_mask.diagonal(offset=0, dim1=-2, dim2=-1)
            intra_diagonal_elements.fill_(1.0)

            behavior_mask_final[:, head_index].masked_fill_(intra_b_mask, 0.0)

        scores = scores + behavior_mask_final

        if key_padding_mask is not None:
            scores.masked_fill_(key_padding_mask.view(B, 1, 1, L), float('-inf'))
        
        all_masked_rows = (scores == float('-inf')).all(dim=-1) # (B, H, L)
        if all_masked_rows.any():
            scores.masked_fill_(all_masked_rows.unsqueeze(-1), 0.0)

        attn_weights = torch.softmax(scores, dim=-1) # (B, H, L, L)
        if all_masked_rows.any():
            attn_weights = attn_weights * ((~all_masked_rows).unsqueeze(-1).float())
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, V) # (B, H, L, L) @ (B, H, L, d_h) -> (B, H, L, d_h)

        gate_score = torch.einsum('bhld,hde->bhle', output, self.gate_weight) + self.gate_bias.unsqueeze(1)
        gate_score = torch.sigmoid(gate_score)
        output = output * gate_score

        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = output.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        output = self.W_o(output)

        return output, attn_weights
    
class BehaviorAwareEncoderLayer(nn.Module):
    def __init__(self, d_model, num_behaviors, head_splits, d_ff, attn_dropout=0.1, dropout=0.1):
        super().__init__()

        self.self_attn = BehaviorAareMHA(d_model, num_behaviors, head_splits, attn_dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, behavior_mask, key_padding_mask=None):
        attn_output, _ = self.self_attn(x, behavior_mask, key_padding_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)

        return x
    
class BAMHA_Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_behaviors, head_splits, d_ff, attn_dropout=0.1, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            BehaviorAwareEncoderLayer(
                d_model, num_behaviors, head_splits,
                d_ff, attn_dropout, dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x, behavior_mask, key_padding_mask=None):
        output = x
        for layer in self.layers:
            output = layer(output, behavior_mask, key_padding_mask)
        
        return output
    
class MCA(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.h_d = d_model // n_head
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, target_seq, memory_seq, target_key_padding_mask, memory_key_padding_mask):
        batch_size = target_seq.shape[0]

        # [B, H, L, d_h]
        Q = self.W_Q(target_seq).view(batch_size, -1, self.n_head, self.h_d).transpose(1, 2)
        K = self.W_K(memory_seq).view(batch_size, -1, self.n_head, self.h_d).transpose(1, 2)
        V = self.W_V(memory_seq).view(batch_size, -1, self.n_head, self.h_d).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.h_d ** 0.5) # [B, H, L, h_d] @ [B, H, h_d, L] = [B, H, L, L]
        scores = scores.masked_fill(memory_key_padding_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        context = attn_weights @ V # [B, H, L, L] @ [B, H, L, d_h] = [B, H, L, d_h]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # [B, L, D]
        context = context.masked_fill(target_key_padding_mask.unsqueeze(-1) == 0, 0.0)
        output = self.W_O(context)

        return output

class OrdinalLogitRegression(nn.Module):
    def __init__(self, margin_1, margin_2, normalize=True, device='cuda:0'):
        super().__init__()
        self.normalize = normalize
        self.margin_1 = nn.Parameter(torch.tensor(margin_1))
        self.margin_2 = nn.Parameter(torch.tensor(margin_2))
        self.softplus = nn.Softplus()
        self.to(device)
    
    def forward(self, z_q, z_c, z_u_list, u_mask, z_n_list):
        B, K, D = z_u_list.shape
        M = z_n_list.shape[1]
        if self.normalize:
            z_q = F.normalize(z_q, p=2, dim=1)
            z_c = F.normalize(z_c, p=2, dim=1)
            z_u_list = F.normalize(z_u_list, p=2, dim=2)
            z_n_list = F.normalize(z_n_list, p=2, dim=2)

        s_c = torch.sum(z_q * z_c, dim=1)
        s_u = torch.bmm(z_q.unsqueeze(1), z_u_list.transpose(1, 2)).squeeze(1)
        s_n = torch.bmm(z_q.unsqueeze(1), z_n_list.transpose(1, 2)).squeeze(1)

        L_c = - torch.log(torch.sigmoid(s_c - (self.margin_1 + self.softplus(self.margin_2)))).sum() / B
        L_u = - (torch.log(torch.sigmoid((self.margin_1 + self.softplus(self.margin_2)) - s_u) - torch.sigmoid(self.margin_1 - s_u) + 1e-8) * u_mask).sum() / u_mask.sum()
        L_n = - torch.log(torch.sigmoid(self.margin_1 - s_n)).sum() / (B * M)

        total_loss = L_c + L_u + L_n
        return total_loss
