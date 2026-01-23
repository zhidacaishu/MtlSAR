import torch
import torch.nn as nn
import torch.nn.functional as F
from models import const, layers, inputs
from utils import util
import os
import logging

class MtlSAR(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--model_path', type=str, default='')
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--hid_dropout', type=float, default=0.1)
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--num_heads', type=int, default=4)

        parser.add_argument('--q_i_cl_weight', type=float, default=1e-3)

        parser.add_argument('--margin_1', type=float, default=0.1)
        parser.add_argument('--margin_2', type=float, default=-2.0)
        
        parser.add_argument('--s_r_cl_temp', type=float, default=0.5)
        parser.add_argument('--s_r_cl_weight', type=float, default=1e-2)

        parser.add_argument('--pred_hid_units', type=list, default=[200, 80, 1])

        return parser
    
    def __init__(self, args):
        super(MtlSAR, self).__init__()
        self.device = args.device
        self.model_path = args.model_path
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.batch_size = args.batch_size
        self.dropout = args.dropout
        self.hid_dropout = args.hid_dropout
        self.pred_hid_units = args.pred_hid_units
        self.unified_dim = const.final_emb_size
        self.global_pos = layers.PositionalEmbedding(const.max_src_session_his_len + const.max_rec_his_len, const.final_emb_size)
        user_map_vocab = util.load_pickle(const.user_map_vocab)
        self.user_map_vocab = {k: torch.tensor(v).to(self.device) for k, v in user_map_vocab.items()}
        item_map_vocab = util.load_pickle(const.item_map_vocab)
        self.item_map_vocab = {k: torch.tensor(v).to(self.device) for k, v in item_map_vocab.items()}
        session_map_vocab = util.load_pickle(const.session_map_vocab)
        self.session_map_vocab = {k: torch.tensor(v).to(self.device) for k, v in session_map_vocab.items()}
        self.query_embedding = inputs.QueryFeat()
        self.session_embedding = inputs.SrcSessionFeat(self.query_embedding, inputs.ItemFeat(self.query_embedding, map_vocab=self.item_map_vocab), inputs.UserFeat(map_vocab=self.user_map_vocab), map_vocab=self.session_map_vocab)
        self.transformer_types = const.rec_inter_behavior + ['search', 'global']

        self.transformer_layer = layers.Transformer_layer(emb_size=self.unified_dim, num_heads=self.num_heads, num_layers=self.num_layers, dropout=self.dropout)
        self.behavior_emb = nn.Sequential(
            nn.Linear((len(const.rec_inter_behavior) + 1) * self.unified_dim, self.unified_dim),
            nn.ReLU()
        )

        self.ba_transformer = layers.BAMHA_Encoder(
            num_layers=self.num_layers, d_model=self.unified_dim,
            num_behaviors=len(const.rec_inter_behavior),
            head_splits={'general': 2, 'inter': 2}, d_ff=4 * self.unified_dim,
            attn_dropout=self.dropout, dropout=self.hid_dropout
        )

        self.msa = layers.Transformer_layer(
            emb_size=self.unified_dim, num_heads=self.num_heads,
            num_layers=self.num_layers, dropout=self.dropout
        )

        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.unified_dim, nhead=self.num_heads, dim_feedforward=self.unified_dim,
            dropout=self.dropout, batch_first=True
        )
        
        self.rec_cross_fusion = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=self.num_layers)
        self.src_cross_fusion = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=self.num_layers)

        self.target_attn_PLE = layers.TargetAttentive_PLE_layer(
            d_model=self.unified_dim, bottom_mlp_dims=[self.unified_dim],
            tower_mlp_dims=[2 * self.unified_dim, self.unified_dim], task_num=2,
            shared_expert_num=4, specific_expert_num=4, dropout=self.dropout
        )

        self.q_i_cl_weight = args.q_i_cl_weight
        self.margin_1 = args.margin_1
        self.margin_2 = args.margin_2
        if self.q_i_cl_weight > 0:
            self.query_item_alignment = True
            self.feature_alignment = layers.OrdinalLogitRegression(self.margin_1, self.margin_2, True, device=self.device)
        else:
            self.query_item_alignment = False
        
        self.s_r_cl_temp = nn.Parameter(torch.tensor(args.s_r_cl_temp))
        self.s_r_cl_weight = args.s_r_cl_weight
        self.src_rec_alignment = (self.s_r_cl_weight > 0)
        self.s_r_loss = layers.infoNCE_loss(temp_init=self.s_r_cl_temp, hdim=self.unified_dim)

        self.rec_fc_layer = layers.MultiLayerPerceptron(input_size=3 * self.unified_dim, hidden_unit=self.pred_hid_units, batch_norm=False, sigmoid=True, activation='relu', dropout=self.hid_dropout)
        self.src_fc_layer = layers.MultiLayerPerceptron(input_size=4 * self.unified_dim, hidden_unit=self.pred_hid_units, batch_norm=False, sigmoid=True, activation='relu', dropout=self.hid_dropout)

        self.behavior_embedding = nn.Parameter(torch.zeros((len(const.rec_inter_behavior) + 1, self.unified_dim)))
        nn.init.xavier_normal_(self.behavior_embedding)

        self.rec_proj = nn.Linear(2 * self.unified_dim, self.unified_dim)
        self.src_proj = nn.Linear(2 * self.unified_dim, self.unified_dim)

        self.rec_cl_proj = nn.Sequential(
            nn.Linear(self.unified_dim, self.unified_dim),
            nn.GELU(),
            nn.Linear(self.unified_dim, self.unified_dim)
        )
        self.src_cl_proj = nn.Sequential(
            nn.Linear(self.unified_dim, self.unified_dim),
            nn.GELU(),
            nn.Linear(self.unified_dim, self.unified_dim)
        )

        self.loss_fn = nn.BCELoss()
        self._init_weights()
        self.to(self.device)
    
    def loss(self, inputs):
        if inputs['search']:
            return self.src_loss(inputs)
        else:
            return self.rec_loss(inputs)
        
    def predict(self, inputs):
        if inputs['search']:
            return self.src_predict(inputs)
        else:
            return self.rec_predict(inputs)
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Embedding):
                continue
    
    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
            model_path = os.path.join(model_path, 'best.pt')
        util.check_dir(model_path)
        logging.info(f'Save model to: {model_path}')
        torch.save(self.state_dict(), model_path)
    
    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
            model_path = os.path.join(model_path, 'best.pt')
        logging.info(f'Load model from: {model_path}')
        self.load_state_dict(torch.load(model_path, map_location=self.device))
    
    def customize_parameters(self):
        no_decay = {'bias', 'temp', 'margin'}

        params_decay = []
        params_no_decay = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if any(nd in name for nd in no_decay):
                params_no_decay.append(param)
            else:
                params_decay.append(param)

        optimize_dict = [{'params': params_decay}, {'params': params_no_decay, 'weight_decay': 0.0}]
        
        return optimize_dict
    
    def src_feat_process(self, src_feat):
        query_emb, q_click_item_emb, click_item_mask = src_feat

        mean_click_item_emb = torch.sum(q_click_item_emb * click_item_mask.unsqueeze(-1), dim=-2)
        mean_click_item_emb = mean_click_item_emb / torch.max(click_item_mask.sum(-1, keepdim=True), torch.ones_like(click_item_mask.sum(-1, keepdim=True)))
        
        return query_emb + mean_click_item_emb

    def get_his_emb(self, all_his, all_his_type):
        behavior_mask = {}
        for i, t in enumerate(self.transformer_types):
            if t == 'global':
                behavior_mask[t] = (all_his_type.sum(dim=-1) > 0)
            else:
                behavior_mask[t] = all_his_type[:, :, i]
            if t == 'search':
                src_his = torch.masked_fill(all_his, behavior_mask[t] != 1, 0)
                src_his_emb = self.src_feat_process(self.session_embedding(src_his))
                src_his_emb = torch.masked_fill(src_his_emb, (behavior_mask[t] != 1).unsqueeze(-1), 0)
        behavior_mask['rec'] = torch.zeros_like(behavior_mask['global'])
        for _ in const.rec_inter_behavior:
            behavior_mask['rec'] = behavior_mask['rec'] | behavior_mask[_]
        behavior_mask['rec'] = torch.masked_fill(behavior_mask['rec'], behavior_mask['rec'] != 0, 1)
        rec_his = torch.masked_fill(all_his, behavior_mask['rec'] != 1, 0)
        rec_his_emb = self.session_embedding.get_item_emb(rec_his)
        rec_his_emb = torch.masked_fill(rec_his_emb, (behavior_mask['rec'] == 0).unsqueeze(-1), 0)
        his_emb = rec_his_emb + src_his_emb

        return his_emb, behavior_mask
    
    def split_rec_src(self, all_his_emb, all_his_type):
        rec_his_emb = torch.masked_select(all_his_emb, (all_his_type == 1).unsqueeze(-1)).reshape((all_his_emb.shape[0], const.max_rec_his_len, all_his_emb.shape[2]))
        src_his_emb = torch.masked_select(all_his_emb, (all_his_type == 2).unsqueeze(-1)).reshape((all_his_emb.shape[0], const.max_src_session_his_len, all_his_emb.shape[2]))
        return rec_his_emb, src_his_emb

    def forward(self, user, all_his, all_his_type, items_emb, domain, query_emb=None):
        user_emb = self.session_embedding.get_user_emb(user)
        his_emb, behavior_mask = self.get_his_emb(all_his, all_his_type)

        behavior_one_hot = torch.stack([behavior_mask[b] for b in const.rec_inter_behavior + ['search']], dim=-1)
        his_behavior_emb = behavior_one_hot.float() @ self.behavior_embedding
        all_his_emb = his_emb + self.global_pos(his_emb) + his_behavior_emb

        global_mask = behavior_mask['global']

        rec_enc = self.ba_transformer(all_his_emb * behavior_mask['rec'].unsqueeze(-1), behavior_one_hot[:, :, :-1], ~behavior_mask['rec'].bool())
        src_enc = self.msa(all_his_emb * behavior_mask['search'].unsqueeze(-1), src_key_padding_mask=behavior_mask['search'])

        global_output_tmp = rec_enc * behavior_mask['rec'].unsqueeze(-1) + src_enc * behavior_mask['search'].unsqueeze(-1)
        
        rec_output = rec_enc + self.rec_cross_fusion(tgt=rec_enc, memory=src_enc, tgt_key_padding_mask=~behavior_mask['rec'].bool(), memory_key_padding_mask=~behavior_mask['search'].bool())
        src_output = src_enc + self.src_cross_fusion(tgt=src_enc, memory=rec_enc, tgt_key_padding_mask=~behavior_mask['search'].bool(), memory_key_padding_mask=~behavior_mask['rec'].bool())

        rec_output_clean = rec_output * behavior_mask['rec'].unsqueeze(-1)
        src_output_clean = src_output * behavior_mask['search'].unsqueeze(-1)

        global_output = rec_output_clean + src_output_clean

        if domain == 'src':
            target_emb = items_emb + query_emb.unsqueeze(1) + user_emb.unsqueeze(1)
        else:
            target_emb = items_emb + user_emb.unsqueeze(1)

        rec_interest, src_interest = self.target_attn_PLE(global_output, target_emb, global_mask)

        user_feats = [rec_interest, src_interest, user_emb]
        return user_feats, global_output_tmp, behavior_mask
    
    def inter_pred(self, user_feats, items_emb, domain, query_emb=None):
        rec_interest, src_interest, user_emb = user_feats
        num_items_per_user = items_emb.shape[1]
        items_emb_flat = items_emb.reshape(-1, items_emb.shape[-1])
        user_emb_flat = user_emb.unsqueeze(1).repeat(1, num_items_per_user, 1).reshape(-1, user_emb.shape[-1])

        if domain == 'rec':
            rec_interest_flat = rec_interest.reshape(-1, rec_interest.shape[-1])
            output = torch.cat([rec_interest_flat, user_emb_flat, items_emb_flat], dim=-1)
            return self.rec_fc_layer(output)
        else:
            src_interest_flat = src_interest.reshape(-1, src_interest.shape[-1])
            query_emb_flat = query_emb.unsqueeze(1).repeat(1, num_items_per_user, 1).reshape(-1, query_emb.shape[-1])
            output = torch.cat([src_interest_flat, user_emb_flat, items_emb_flat, query_emb_flat], dim=-1)
            return self.src_fc_layer(output)

    def rec_predict(self, inputs):
        user, all_his, all_his_type, pos_item, neg_items = inputs['user'], inputs['all_his'], inputs['all_his_type'], inputs['item'], inputs['neg_items']
        behavior_type = inputs['behavior_type']

        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        items_emb = self.session_embedding.get_item_emb(items)
        batch_size = items_emb.size(0)

        user_feats, _, _ = self.forward(user, all_his, all_his_type, items_emb, domain='rec')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        batch_indices = torch.arange(batch_size, device=all_his_type.device)
        click_onehot = behavior_type[batch_indices, 0].bool()
        logits = self.inter_pred(user_feats, items_emb, domain='rec').reshape((batch_size, -1))[click_onehot]
        
        return logits
    
    def src_predict(self, inputs):
        user, all_his, all_his_type, pos_item, neg_items = inputs['user'], inputs['all_his'], inputs['all_his_type'], inputs['item'], inputs['neg_items']
        query = inputs['query']
        query_emb = self.session_embedding.get_query_emb(query)

        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        items_emb = self.session_embedding.get_item_emb(items)
        batch_size = items_emb.size(0)

        user_feats, _, _ = self.forward(user, all_his, all_his_type, items_emb, domain='src', query_emb=query_emb)

        logits = self.inter_pred(user_feats, items_emb, domain='src', query_emb=query_emb).reshape((batch_size, -1))
        return logits

    def rec_loss(self, inputs):
        user, all_his, all_his_type, pos_item, neg_items = inputs['user'], inputs['all_his'], inputs['all_his_type'], inputs['item'], inputs['neg_items']
        behavior_type = inputs['behavior_type']

        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        items_emb = self.session_embedding.get_item_emb(items)
        batch_size = items_emb.size(0)

        user_feats, _, _ = self.forward(user, all_his, all_his_type, items_emb, domain='rec')

        batch_indices = torch.arange(batch_size, device=all_his_type.device)
        click_onehot = behavior_type[batch_indices, 0].float()
        logits = self.inter_pred(user_feats, items_emb, domain='rec').reshape((batch_size, -1))
        labels = torch.zeros_like(logits, dtype=torch.float32)
        labels[:, 0] = click_onehot

        logits = logits.reshape((-1, ))
        labels = labels.reshape((-1, ))

        total_loss = self.loss_fn(logits, labels)
        loss_dict = {}
        loss_dict['click_loss'] = total_loss.clone()

        loss_dict['total_loss'] = total_loss.clone()

        return loss_dict
    
    def src_loss(self, inputs):
        user, all_his, all_his_type, pos_item, neg_items = inputs['user'], inputs['all_his'], inputs['all_his_type'], inputs['item'], inputs['neg_items']
        query, weak_neg_items = inputs['query'], inputs['weak_neg_items']
        weak_neg_items_mask = (weak_neg_items > 0)
        query_emb = self.session_embedding.get_query_emb(query)
        
        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        items_emb = self.session_embedding.get_item_emb(items)

        pos_items_emb = self.session_embedding.get_item_emb(pos_item)
        weak_neg_items_emb = self.session_embedding.get_item_emb(weak_neg_items)
        neg_items_emb = self.session_embedding.get_item_emb(neg_items)
        batch_size = items_emb.size(0)

        user_feats, global_output_tmp, behavior_mask = self.forward(user, all_his, all_his_type, items_emb, domain='src', query_emb=query_emb)

        logits = self.inter_pred(user_feats, items_emb, domain='src', query_emb=query_emb).reshape((batch_size, -1))
        labels = torch.zeros_like(logits, dtype=torch.float32)
        labels[:, 0] = 1.0

        logits = logits.reshape((-1, ))
        labels = labels.reshape((-1, ))
        
        total_loss = self.loss_fn(logits, labels)
        loss_dict = {}
        loss_dict['click_loss'] = total_loss.clone()

        if self.q_i_cl_weight > 0:
            align_loss = self.feature_alignment(query_emb, pos_items_emb, weak_neg_items_emb, weak_neg_items_mask, neg_items_emb)

            loss_dict['q_i_cl_loss'] = align_loss.clone()
            total_loss += self.q_i_cl_weight * align_loss

        if self.s_r_cl_weight > 0:
            h_rec = (global_output_tmp * behavior_mask['rec'].unsqueeze(-1).float()).sum(dim=1) / behavior_mask['rec'].sum(dim=1, keepdim=True)
            h_src = (global_output_tmp * behavior_mask['search'].unsqueeze(-1).float()).sum(dim=1) / behavior_mask['search'].sum(dim=1, keepdim=True)
            h_rec_neg = torch.roll(h_rec, shifts=1, dims=0)
            h_src_neg = torch.roll(h_src, shifts=1, dims=0)

            h_rec = self.rec_cl_proj(h_rec)
            h_src = self.src_cl_proj(h_src)
            h_rec_neg = self.rec_cl_proj(h_rec_neg)
            h_src_neg = self.src_cl_proj(h_src_neg)

            cl_loss = self.s_r_loss(h_src, h_rec, h_rec_neg, h_src_neg)

            loss_dict['s_r_cl_loss'] = cl_loss.clone()
            total_loss += self.s_r_cl_weight * cl_loss

        loss_dict['total_loss'] = total_loss.clone()

        return loss_dict


