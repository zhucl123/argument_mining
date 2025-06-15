"""BERT and RNN model for sentence pair classification.

Author: Yixu GAO (yxgao19@fudan.edu.cn)

Used for SMP-CAIL2020-Argmine.
"""
import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from transformers import BertModel, AutoModel
from transformers import AutoTokenizer, AutoModelForMaskedLM


# baseline v1.0
class BertForClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.linear = nn.Linear(config.hidden_size, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        self.num_classes = config.num_classes

    def forward(self, input_ids, attention_mask, token_type_ids):

        batch_size = input_ids.shape[0]
        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_states=False,
            return_dict=True)

        pooled_output = bert_output.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits

import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
# from utils import get_vocab


class BertClassifierv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        
        self.hidden_size = config.hidden_size
        self.num_classes = config.num_classes

        # 词对齐相关组件
        self.alignment_projector = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128)
        )
        
        # 对齐注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=config.dropout
        )
        
        # 更新特征融合层的输入维度
        self.feature_fusion = nn.Sequential(
            nn.Linear(256 * 5 + 128, hidden_size),  # 增加对齐特征维度
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Tanh()
        )
        # 注意力机制 - 针对动作词和否定词
        self.action_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size, 
            num_heads=8, 
            dropout=config.dropout
        )
        self.negation_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size, 
            num_heads=8, 
            dropout=config.dropout
        )
        
        # 特征融合层
        # pooled_output + sentence_diff + attention_features
        feature_dim = self.hidden_size * 4  # pooled + diff + action_attn + neg_attn
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Tanh()
        )
        
        # self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 128),  # 4种特征融合
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, config.num_classes)
        )
        self.dropout = nn.Dropout(config.dropout)
        
        # 动作词和否定词的词汇表（可以根据需要扩展）
        self.action_words = get_vocab('verb.txt')
        self.negation_words = {'不', '没', '没有', '无', '勿', '非', '未', '否'}
    
    def get_word_mask(self, input_ids, tokenizer, word_set):
        """获取特定词汇的mask"""
        batch_size, seq_len = input_ids.shape
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)
        
        for i in range(batch_size):
            tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
            for j, token in enumerate(tokens):
                # 处理subword tokens
                clean_token = token.replace('##', '').lower()
                if clean_token in word_set:
                    mask[i, j] = True
        
        return mask
    
    def apply_attention(self, hidden_states, attention_layer, mask=None):
        """应用注意力机制"""
        # hidden_states: (batch_size, seq_len, hidden_size)
        seq_len = hidden_states.size(1)
        
        # 转换为 (seq_len, batch_size, hidden_size) 用于MultiheadAttention
        hidden_states_t = hidden_states.transpose(0, 1)
        
        if mask is not None:
            # 创建attention mask
            attn_mask = ~mask  # MultiheadAttention中True表示忽略
        else:
            attn_mask = None
        
        # 应用注意力
        attn_output, attn_weights = attention_layer(
            hidden_states_t, hidden_states_t, hidden_states_t,
            key_padding_mask=attn_mask
        )
        
        # 转换回 (batch_size, seq_len, hidden_size)
        attn_output = attn_output.transpose(0, 1)
        
        # 全局池化
        if mask is not None:
            # 只对有效位置进行平均
            mask_expanded = mask.unsqueeze(-1).expand_as(attn_output)
            masked_output = attn_output * mask_expanded.float()
            pooled = masked_output.sum(dim=1) / (mask.sum(dim=1, keepdim=True).float() + 1e-8)
        else:
            pooled = attn_output.mean(dim=1)
        
        return pooled
    
    def compute_sentence_diff(self, hidden_states, input_ids, sep_token_id=102):
        """计算句对的向量差"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        sentence_diffs = []
        
        for i in range(batch_size):
            # 找到[SEP]token的位置来分割两个句子
            sep_positions = (input_ids[i] == sep_token_id).nonzero().flatten()
            
            if len(sep_positions) >= 2:
                # 第一个句子: [CLS] ... [SEP]
                sent1_end = sep_positions[0]
                # 第二个句子: [SEP] ... [SEP]
                sent2_start = sep_positions[0] + 1
                sent2_end = sep_positions[1] if len(sep_positions) > 1 else seq_len
                
                # 计算句子向量（平均池化）
                sent1_vec = hidden_states[i, 1:sent1_end].mean(dim=0)  # 排除[CLS]
                sent2_vec = hidden_states[i, sent2_start:sent2_end].mean(dim=0)
                
                # 计算向量差
                sent_diff = torch.abs(sent1_vec - sent2_vec)
            else:
                # 如果没有找到合适的[SEP]，使用零向量
                sent_diff = torch.zeros(hidden_size, device=hidden_states.device)
            
            sentence_diffs.append(sent_diff)
        
        return torch.stack(sentence_diffs)
    
    def forward(self, input_ids, attention_mask, token_type_ids, tokenizer=None):

        batch_size = input_ids.shape[0]

        # BERT编码
        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True)
        
        # 获取序列输出和池化输出
        sequence_output = bert_output.last_hidden_state  # (batch_size, seq_len, hidden_size)
        pooled_output = bert_output.pooler_output        # (batch_size, hidden_size)
        
        # 1. 基础池化特征
        pooled_features = self.dropout(pooled_output)
        
        # 2. 句对向量差特征
        sentence_diff = self.compute_sentence_diff(sequence_output, input_ids)
        
        # 3. 注意力特征
        if tokenizer is not None:
            # 获取动作词和否定词的mask
            action_mask = self.get_word_mask(input_ids, tokenizer, self.action_words)
            negation_mask = self.get_word_mask(input_ids, tokenizer, self.negation_words)
            
            # 应用注意力机制
            action_features = self.apply_attention(sequence_output, self.action_attention, action_mask)
            negation_features = self.apply_attention(sequence_output, self.negation_attention, negation_mask)
        else:
            # 如果没有tokenizer，使用全局注意力
            action_features = self.apply_attention(sequence_output, self.action_attention)
            negation_features = self.apply_attention(sequence_output, self.negation_attention)
        
        # 特征拼接
        combined_features = torch.cat([
            pooled_features,      # 基础BERT特征
            sentence_diff,        # 句对差异特征
            action_features,      # 动作词注意力特征
            negation_features     # 否定词注意力特征
        ], dim=-1)
        
        # 特征融合
        fused_features = self.feature_fusion(combined_features)
        # print(fused_features.shape)
        # 分类
        logits = self.classifier(fused_features)
        
        # 注意：通常不在模型内部应用softmax，而是在loss计算时使用
        # 如果需要概率输出，可以在推理时再应用
        return logits


class BertClassifierv3(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # BERT编码器
        self.bert = AutoModel.from_pretrained(config.bert_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_model_path)
        self.dropout = nn.Dropout(config.dropout)
        
        # 特征提取层
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256)
        )

        # 句子表示提取
        # self.sentence_pooler = nn.Sequential(
        #     nn.Linear(hidden_size, 256),
        #     nn.Tanh()
        # )
        # 孪生网络
        self.cls_projector = nn.Sequential(nn.Linear(hidden_size, 256), nn.Tanh())
        self.mean_projector = nn.Sequential(nn.Linear(hidden_size, 256), nn.Tanh())
        self.max_projector = nn.Sequential(nn.Linear(hidden_size, 256), nn.Tanh())
        self.diff_projector = nn.Sequential(nn.Linear(hidden_size, 256), nn.Tanh())
        # self.act_projector = nn.Sequential(nn.Linear(hidden_size, 256), nn.Tanh())
        # self.neg_projector = nn.Sequential(nn.Linear(hidden_size, 256), nn.Tanh())

        # 动作词和否定词的词汇表（可以根据需要扩展）
        # self.action_words = get_vocab('verb.txt') | {'不', '没', '没有', '无', '勿', '非', '未', '否'}
        # self.negation_words = {'不', '没', '没有', '无', '勿', '非', '未', '否'}

        # # 注意力机制 - 针对动作词和否定词
        # self.action_attention = nn.MultiheadAttention(
        #     embed_dim=hidden_size, 
        #     num_heads=8, 
        #     dropout=config.dropout
        # )
        # self.negation_attention = nn.MultiheadAttention(
        #     embed_dim=hidden_size, 
        #     num_heads=8, 
        #     dropout=config.dropout
        # )
        self.feature_fusion = nn.Sequential(
            nn.Linear(256 * 5, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Tanh()
        )
        
        # 修改分类器输入维度
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),  # 4种特征融合
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, config.num_classes)
        )
        
        # 注意力权重（用于可解释性）
        self.attention_weights = None
    
    def extract_sentence_representations(self, sequence_output, attention_mask):
        """提取句子级别的表示"""
        # 使用attention mask进行平均池化
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    def masked_max_pooling(self, sequence_output, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(sequence_output.size())
        # 将无效位置设为极小值（-inf 近似）
        sequence_output = sequence_output.masked_fill(mask == 0, -1e9)
        return torch.max(sequence_output, dim=1)[0]

    def compute_sentence_diff(self, hidden_states, input_ids, sep_token_id=102):
        batch_size, seq_len, hidden_size = hidden_states.shape
        sentence_diffs = []
        
        for i in range(batch_size):
            # 找到[SEP]token的位置来分割两个句子
            sep_positions = (input_ids[i] == sep_token_id).nonzero().flatten()
            
            if len(sep_positions) >= 2:
                # 第一个句子: [CLS] ... [SEP]
                sent1_end = sep_positions[0]
                # 第二个句子: [SEP] ... [SEP]
                sent2_start = sep_positions[0] + 1
                sent2_end = sep_positions[1] if len(sep_positions) > 1 else seq_len
                
                # 计算句子向量（平均池化）
                sent1_vec = hidden_states[i, 1:sent1_end].mean(dim=0)  # 排除[CLS]
                sent2_vec = hidden_states[i, sent2_start:sent2_end].mean(dim=0)
                
                # 计算向量差
                sent_diff = torch.abs(sent1_vec - sent2_vec)
            else:
                # 如果没有找到合适的[SEP]，使用零向量
                sent_diff = torch.zeros(hidden_size, device=hidden_states.device)
            
            sentence_diffs.append(sent_diff)
        
        return torch.stack(sentence_diffs)

    def get_word_mask(self, input_ids, word_set):
        """获取特定词汇的mask"""
        batch_size, seq_len = input_ids.shape
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)
        
        for i in range(batch_size):
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
            for j, token in enumerate(tokens):
                # 处理subword tokens
                clean_token = token.replace('##', '').lower()
                if clean_token in word_set:
                    mask[i, j] = True
        
        return mask
    
    def apply_attention(self, hidden_states, attention_layer, mask=None):
        """应用注意力机制"""
        # hidden_states: (batch_size, seq_len, hidden_size)
        seq_len = hidden_states.size(1)
        
        # 转换为 (seq_len, batch_size, hidden_size) 用于MultiheadAttention
        hidden_states_t = hidden_states.transpose(0, 1)
        
        if mask is not None:
            # 创建attention mask
            attn_mask = ~mask  # MultiheadAttention中True表示忽略
        else:
            attn_mask = None
        
        # 应用注意力
        attn_output, attn_weights = attention_layer(
            hidden_states_t, hidden_states_t, hidden_states_t,
            key_padding_mask=attn_mask
        )
        
        # 转换回 (batch_size, seq_len, hidden_size)
        attn_output = attn_output.transpose(0, 1)
        
        # 全局池化
        if mask is not None:
            # 只对有效位置进行平均
            mask_expanded = mask.unsqueeze(-1).expand_as(attn_output)
            masked_output = attn_output * mask_expanded.float()
            pooled = masked_output.sum(dim=1) / (mask.sum(dim=1, keepdim=True).float() + 1e-8)
        else:
            pooled = attn_output.mean(dim=1)
        
        return pooled
    
    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True
        )
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_length, hidden_size]
        pooled_output = sequence_output[:, 0]        # [batch_size, hidden_size]
        
        # 保存注意力权重用于可解释性分析
        # self.attention_weights = outputs.attentions
        
        # 方法1：使用[CLS]标记的表示
        cls_representation = self.cls_projector(pooled_output)
        
        # 方法2：平均池化获取句子表示
        mean_representation = self.extract_sentence_representations(sequence_output, attention_mask)
        mean_representation = self.mean_projector(mean_representation)
        
        # 方法3：最大池化
        max_representation = self.masked_max_pooling(sequence_output, attention_mask)
        max_representation = self.max_projector(max_representation)
        
        # 方法4：特征提取器处理的表示
        feature_representation = self.feature_extractor(pooled_output)

        # 方法5: 句对交互特征
        sentence_diff = self.compute_sentence_diff(sequence_output, input_ids)
        sentence_diff = self.diff_projector(sentence_diff)

        # 方法6: 注意力特征: 动作词和否定词
        ## 获取动作词和否定词的mask
        # action_mask = self.get_word_mask(input_ids, self.action_words)
        # negation_mask = self.get_word_mask(input_ids, self.negation_words)
        
        # ## 应用注意力机制
        # action_features = self.apply_attention(sequence_output, self.action_attention, action_mask)
        # negation_features = self.apply_attention(sequence_output, self.negation_attention, negation_mask)
        
        # act_atten = self.act_projector(action_features * pooled_output)
        # neg_atten = self.neg_projector(negation_features * pooled_output)
        
        # 特征融合
        combined_features = torch.cat([
            cls_representation,     # 基础BERT特征
            mean_representation,
            max_representation,
            feature_representation,
            sentence_diff,        # 句对差异特征
            # act_atten      # 动作词注意力特征
            # negation_features     # 否定词注意力特征
        ], dim=-1)
        
        # 特征融合
        fused_features = self.feature_fusion(combined_features)
        # 分类
        logits = self.classifier(fused_features)

        return logits

class TeLU(nn.Module):
    def __init__(self):
        super(TeLU, self).__init__()
    
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))

class BertClassifierv4(nn.Module):
    def __init__(self, config, glove_embeddings=None):
        super().__init__()
        
        # BERT编码器
        self.bert = AutoModel.from_pretrained(config.bert_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_model_path)
        self.dropout = nn.Dropout(config.dropout)
        
        # 特征提取层
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            
            nn.LayerNorm(512),
            nn.Dropout(config.dropout),
            nn.Linear(512, 128),
            TeLU()
        )

        # 词对齐相关组件
        self.alignment_projector = nn.Sequential(
            nn.Linear(hidden_size * 2, 768),
            # nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(768, 128),
            TeLU()
        )
        
        # 对齐注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=config.dropout
        )

        # 孪生网络
        # self.cls_projector = nn.Sequential(nn.Linear(hidden_size, 256), TeLU())
        # self.mean_projector = nn.Sequential(nn.Linear(hidden_size, 256), TeLU())
        # self.max_projector = nn.Sequential(nn.Linear(hidden_size, 256), TeLU())
        self.diff_projector = nn.Sequential(nn.Linear(hidden_size, 128), TeLU())
        self.attention_projector = nn.Sequential(nn.Linear(hidden_size, 128), TeLU())
        
        # 修改分类器输入维度
        self.classifier = nn.Sequential(
            # nn.Linear(256 * 4, 256 * 1),  # 4种特征融合
            # nn.ReLU(),
            # nn.Dropout(config.dropout),
            nn.Linear(128 * 4, 64),
            TeLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.num_classes)
        )
        
        # 注意力权重（用于可解释性）
        self.attention_weights = None


    def compute_word_alignment(self, hidden_states, input_ids, attention_mask):
        """
        计算前提和假设之间的词级别对齐
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        sep_token_id=self.tokenizer.sep_token_id
        alignment_features = []
        
        for i in range(batch_size):
            # 找到[SEP]token位置分割前提和假设
            sep_positions = (input_ids[i] == sep_token_id).nonzero().flatten()
            
            if len(sep_positions) >= 2:
                # 前提: [CLS] ... [SEP]
                premise_start, premise_end = 1, sep_positions[0]
                # 假设: [SEP] ... [SEP] 
                hypothesis_start = sep_positions[0] + 1
                hypothesis_end = sep_positions[1] if len(sep_positions) > 1 else seq_len
                
                # 提取前提和假设的隐藏状态
                premise_hidden = hidden_states[i, premise_start:premise_end]  # [premise_len, hidden_size]
                hypothesis_hidden = hidden_states[i, hypothesis_start:hypothesis_end]  # [hyp_len, hidden_size]
                
                # 计算前提到假设的对齐
                premise_to_hyp_alignment = self.compute_alignment_scores(
                    premise_hidden, hypothesis_hidden
                )
                
                # 计算假设到前提的对齐  
                hyp_to_premise_alignment = self.compute_alignment_scores(
                    hypothesis_hidden, premise_hidden
                )
                
                # 聚合对齐特征
                alignment_feature = torch.cat([
                    premise_to_hyp_alignment.mean(dim=0),  # 前提->假设的平均对齐
                    hyp_to_premise_alignment.mean(dim=0)   # 假设->前提的平均对齐
                ])
                
            else:
                # 如果分割失败，使用零向量
                alignment_feature = torch.zeros(hidden_size * 2, device=hidden_states.device)
            
            alignment_features.append(alignment_feature)
        
        return torch.stack(alignment_features)
    
    def compute_alignment_scores(self, source_hidden, target_hidden):
        """
        计算源序列到目标序列的对齐分数
        """
        # source_hidden: [source_len, hidden_size]
        # target_hidden: [target_len, hidden_size]
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(source_hidden, target_hidden.transpose(-2, -1))
        # [source_len, target_len]
        
        # 应用softmax得到对齐权重
        alignment_weights = torch.softmax(similarity_matrix, dim=-1)
        
        # 基于对齐权重计算加权目标表示
        aligned_representations = torch.matmul(alignment_weights, target_hidden)
        # [source_len, hidden_size]
        
        return aligned_representations
    
    def extract_cross_attention_features(self, hidden_states, input_ids, attention_mask):
        """
        使用交叉注意力提取前提-假设交互特征
        """
        batch_size = hidden_states.shape[0]
        sep_token_id=self.tokenizer.sep_token_id
        cross_features = []
        
        for i in range(batch_size):
            sep_positions = (input_ids[i] == sep_token_id).nonzero().flatten()
            
            if len(sep_positions) >= 2:
                premise_start, premise_end = 1, sep_positions[0]
                hypothesis_start = sep_positions[0] + 1  
                hypothesis_end = sep_positions[1] if len(sep_positions) > 1 else hidden_states.shape[1]
                
                premise_hidden = hidden_states[i:i+1, premise_start:premise_end]  # [1, premise_len, hidden_size]
                hypothesis_hidden = hidden_states[i:i+1, hypothesis_start:hypothesis_end]  # [1, hyp_len, hidden_size]
                
                # 转换维度用于MultiheadAttention: [seq_len, batch_size, hidden_size]
                premise_t = premise_hidden.transpose(0, 1)  # [premise_len, 1, hidden_size]
                hypothesis_t = hypothesis_hidden.transpose(0, 1)  # [hyp_len, 1, hidden_size]
                
                # 前提作为query，假设作为key和value
                cross_attn_output, _ = self.cross_attention(
                    hypothesis_t,  # query（第二个句子：被告）
                    premise_t,     # key
                    premise_t      # value
                )
                
                # 平均池化得到交叉特征
                cross_feature = cross_attn_output.mean(dim=0).squeeze(0)  # [hidden_size]
                
            else:
                cross_feature = torch.zeros(hidden_states.shape[-1], device=hidden_states.device)
            
            cross_features.append(cross_feature)
        
        return torch.stack(cross_features)
    def extract_sentence_representations(self, sequence_output, attention_mask):
        """提取句子级别的表示"""
        # 使用attention mask进行平均池化
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    def masked_max_pooling(self, sequence_output, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(sequence_output.size())
        # 将无效位置设为极小值（-inf 近似）
        sequence_output = sequence_output.masked_fill(mask == 0, -1e9)
        return torch.max(sequence_output, dim=1)[0]

    def compute_sentence_diff(self, hidden_states, input_ids):
        sep_token_id=self.tokenizer.sep_token_id
        batch_size, seq_len, hidden_size = hidden_states.shape
        sentence_diffs = []
        
        for i in range(batch_size):
            # 找到[SEP]token的位置来分割两个句子
            sep_positions = (input_ids[i] == sep_token_id).nonzero().flatten()
            
            if len(sep_positions) >= 2:
                # 第一个句子: [CLS] ... [SEP]
                sent1_end = sep_positions[0]
                # 第二个句子: [SEP] ... [SEP]
                sent2_start = sep_positions[0] + 1
                sent2_end = sep_positions[1] if len(sep_positions) > 1 else seq_len
                
                # 计算句子向量（平均池化）
                sent1_vec = hidden_states[i, 1:sent1_end].mean(dim=0)
                sent2_vec = hidden_states[i, sent2_start:sent2_end].mean(dim=0)
                
                # 计算向量差
                sent_diff = torch.abs(sent1_vec - sent2_vec)
            else:
                # 如果没有找到合适的[SEP]，使用零向量
                sent_diff = torch.zeros(hidden_size, device=hidden_states.device)
            
            sentence_diffs.append(sent_diff)
        
        return torch.stack(sentence_diffs)

    def get_word_mask(self, input_ids, word_set):
        """获取特定词汇的mask"""
        batch_size, seq_len = input_ids.shape
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)
        
        for i in range(batch_size):
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
            for j, token in enumerate(tokens):
                clean_token = token.replace('##', '').lower()
                if clean_token in word_set:
                    mask[i, j] = True
        
        return mask
    
    def apply_attention(self, hidden_states, attention_layer, mask=None):
        """应用注意力机制"""
        # hidden_states: (batch_size, seq_len, hidden_size)
        seq_len = hidden_states.size(1)
        
        # 转换为 (seq_len, batch_size, hidden_size) 用于MultiheadAttention
        hidden_states_t = hidden_states.transpose(0, 1)
        
        if mask is not None:
            # 创建attention mask
            attn_mask = ~mask  # MultiheadAttention中True表示忽略
        else:
            attn_mask = None
        
        # 应用注意力
        attn_output, attn_weights = attention_layer(
            hidden_states_t, hidden_states_t, hidden_states_t,
            key_padding_mask=attn_mask
        )
        
        # 转换回 (batch_size, seq_len, hidden_size)
        attn_output = attn_output.transpose(0, 1)
        
        # 全局池化
        if mask is not None:
            # 只对有效位置进行平均
            mask_expanded = mask.unsqueeze(-1).expand_as(attn_output)
            masked_output = attn_output * mask_expanded.float()
            pooled = masked_output.sum(dim=1) / (mask.sum(dim=1, keepdim=True).float() + 1e-8)
        else:
            pooled = attn_output.mean(dim=1)
        
        return pooled
    
    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True
        )
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_length, hidden_size]
        pooled_output = sequence_output[:, 0]        # [batch_size, hidden_size]
        
        # 保存注意力权重用于可解释性分析
        # self.attention_weights = outputs.attentions
        
        # 方法1：使用[CLS]标记的表示
        # cls_representation = self.cls_projector(pooled_output)
        
        # 方法2：平均池化获取句子表示
        mean_representation = self.extract_sentence_representations(sequence_output, attention_mask)
        # mean_representation = self.mean_projector(mean_representation)
        
        # 方法3：最大池化
        max_representation = self.masked_max_pooling(sequence_output, attention_mask)
        max_representation = self.feature_extractor(pooled_output + mean_representation + max_representation)

        # 方法4: 句对差特征
        sentence_diff = self.compute_sentence_diff(sequence_output, input_ids)
        sentence_diff = self.diff_projector(sentence_diff)

        # 方法6: 交叉注意力特征
        cross_attention_features = self.extract_cross_attention_features(sequence_output, input_ids, attention_mask)
        cross_attention_representation = self.attention_projector(cross_attention_features)

        # 方法7: 词级别对齐特征
        alignment_features = self.compute_word_alignment(sequence_output, input_ids, attention_mask)
        alignment_representation = self.alignment_projector(alignment_features)
        
        
        # 特征融合
        combined_features = torch.cat([
            max_representation,     # 基础BERT特征
            sentence_diff,
            cross_attention_representation,
            alignment_representation
        ], dim=-1)
        
        # 分类
        logits = self.classifier(combined_features)

        return logits

import networkx as nx
from collections import defaultdict
import jieba
import jieba.posseg as pseg

class BertClassifierv5(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # BERT编码器
        self.bert = AutoModel.from_pretrained(config.bert_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_model_path)
        self.dropout = nn.Dropout(config.dropout)
        
        # 特征提取层
        hidden_size = self.bert.config.hidden_size
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(config.dropout),
            nn.Linear(512, 128)
        )

        # 孪生网络投影层
        self.cls_projector = nn.Sequential(nn.Linear(hidden_size, 128), nn.Tanh())
        self.mean_projector = nn.Sequential(nn.Linear(hidden_size, 128), nn.Tanh())
        self.max_projector = nn.Sequential(nn.Linear(hidden_size, 128), nn.Tanh())
        self.diff_projector = nn.Sequential(nn.Linear(hidden_size, 128), nn.Tanh())
        self.event_projector = nn.Sequential(nn.Linear(hidden_size, 128), nn.Tanh())

        # TextRank参数
        self.textrank_window = 3  # 共现窗口大小
        self.textrank_iterations = 30  # 迭代次数
        self.top_k_events = 5  # 提取的top-k事件数量
        
        # 事件相关的注意力机制
        self.event_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=8, 
            dropout=config.dropout
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128 * 5, 128 * 3),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        self.attention_weights = None
    
    def extract_events_with_textrank(self, text):
        """使用TextRank提取关键事件词汇"""
        # 分词和词性标注
        words = list(jieba.cut(text))
        pos_words = [(w, flag) for w, flag in pseg.cut(text)]
        
        # 过滤出动作相关词汇（动词、形容词等）
        event_candidates = []
        for word, pos in pos_words:
            if len(word) > 1 and pos in ['v', 'vn', 'vd', 'a', 'ad', 'n', 'nr', 'ns', 'nt']:
                event_candidates.append(word)
        
        if len(event_candidates) < 2:
            return set()
        
        # 构建词汇共现图
        graph = nx.Graph()
        
        # 添加节点
        for word in event_candidates:
            graph.add_node(word)
        
        # 添加边（基于窗口内的共现）
        for i, word1 in enumerate(event_candidates):
            for j in range(max(0, i - self.textrank_window), 
                          min(len(event_candidates), i + self.textrank_window + 1)):
                if i != j:
                    word2 = event_candidates[j]
                    if graph.has_edge(word1, word2):
                        graph[word1][word2]['weight'] += 1
                    else:
                        graph.add_edge(word1, word2, weight=1)
        
        if len(graph.nodes()) == 0:
            return set()
        
        # 运行TextRank算法
        try:
            scores = nx.pagerank(graph, max_iter=self.textrank_iterations, weight='weight')
            
            # 获取top-k关键词
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_events = {word for word, score in sorted_words[:self.top_k_events]}
            
            return top_events
        except:
            # 如果TextRank失败，返回频率最高的词
            word_freq = defaultdict(int)
            for word in event_candidates:
                word_freq[word] += 1
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return {word for word, freq in top_words[:self.top_k_events]}
    
    def get_dynamic_event_mask(self, input_ids):
        """基于TextRank动态获取事件词的mask"""
        batch_size, seq_len = input_ids.shape
        event_masks = []
        
        for i in range(batch_size):
            # 将token ids转换为文本
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
            text = self.tokenizer.convert_tokens_to_string(tokens)
            text = text.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').strip()
            
            # 使用TextRank提取关键事件
            event_words = self.extract_events_with_textrank(text)
            
            # 创建mask
            mask = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)
            
            for j, token in enumerate(tokens):
                # 处理subword tokens
                clean_token = token.replace('##', '')
                
                # 检查是否匹配任何事件词
                for event_word in event_words:
                    if clean_token in event_word or event_word in clean_token:
                        mask[j] = True
                        break
            
            event_masks.append(mask)
        
        return torch.stack(event_masks)
    
    def apply_textrank_attention(self, hidden_states, input_ids):
        """应用基于TextRank的事件注意力"""
        # 获取动态事件mask
        event_mask = self.get_dynamic_event_mask(input_ids)
        
        # 转换为注意力机制需要的格式
        hidden_states_t = hidden_states.transpose(0, 1)
        
        # 创建key_padding_mask（True表示忽略）
        key_padding_mask = ~event_mask
        
        # 应用多头注意力
        try:
            attn_output, attn_weights = self.event_attention(
                hidden_states_t, hidden_states_t, hidden_states_t,
                key_padding_mask=key_padding_mask
            )
            attn_output = attn_output.transpose(0, 1)
        except:
            # 如果注意力计算失败，使用原始hidden states
            attn_output = hidden_states
        
        # 对事件相关的token进行加权池化
        event_mask_expanded = event_mask.unsqueeze(-1).expand_as(attn_output)
        
        # 计算事件相关特征
        if event_mask.sum() > 0:
            masked_output = attn_output * event_mask_expanded.float()
            event_features = masked_output.sum(dim=1) / (event_mask.sum(dim=1, keepdim=True).float() + 1e-8)
        else:
            # 如果没有事件词，使用平均池化
            event_features = attn_output.mean(dim=1)
        
        return event_features
    
    def extract_sentence_representations(self, sequence_output, attention_mask):
        """提取句子级别的表示"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    def masked_max_pooling(self, sequence_output, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(sequence_output.size())
        sequence_output = sequence_output.masked_fill(mask == 0, -1e9)
        return torch.max(sequence_output, dim=1)[0]

    def compute_sentence_diff(self, hidden_states, input_ids, sep_token_id=102):
        """计算句对的向量差"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        sentence_diffs = []
        
        for i in range(batch_size):
            sep_positions = (input_ids[i] == sep_token_id).nonzero().flatten()
            
            if len(sep_positions) >= 2:
                sent1_end = sep_positions[0]
                sent2_start = sep_positions[0] + 1
                sent2_end = sep_positions[1] if len(sep_positions) > 1 else seq_len
                
                sent1_vec = hidden_states[i, 1:sent1_end].mean(dim=0)
                sent2_vec = hidden_states[i, sent2_start:sent2_end].mean(dim=0)
                
                sent_diff = torch.abs(sent1_vec - sent2_vec)
            else:
                sent_diff = torch.zeros(hidden_size, device=hidden_states.device)
            
            sentence_diffs.append(sent_diff)
        
        return torch.stack(sentence_diffs)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True
        )
        
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]
        
        self.attention_weights = outputs.attentions
        
        # 各种表示方法
        cls_representation = self.cls_projector(pooled_output)
        
        # mean_representation = self.extract_sentence_representations(sequence_output, attention_mask)
        # mean_representation = self.mean_projector(mean_representation)
        
        max_representation = self.masked_max_pooling(sequence_output, attention_mask)
        max_representation = self.max_projector(max_representation)
        
        feature_representation = self.feature_extractor(pooled_output)
        
        sentence_diff = self.compute_sentence_diff(sequence_output, input_ids)
        sentence_diff = self.diff_projector(sentence_diff)
        
        # 新的TextRank事件注意力特征
        event_features = self.apply_textrank_attention(sequence_output, input_ids)
        event_representation = self.event_projector(event_features)
        
        # 特征融合
        combined_features = torch.cat([
            cls_representation,
            # mean_representation, 
            max_representation,
            feature_representation,
            sentence_diff,
            event_representation
        ], dim=1)
        
        logits = self.classifier(combined_features)
        return logits
        