# -*- coding: utf-8 -*-
# @File: model.py
# @Author: Pei Yu
# @Date:   2020-07-28

import torch
import torch.nn as nn
from torch.nn import functional as F


# control the embedding layer train or not
def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False


class DoubleGRU(nn.Module):
    def __init__(self, user_emd_size, init_word_emb, emb_train):
        super(DoubleGRU, self).__init__()
        # news_embedding_size == user_embedding_size
        self.word_emb_size = init_word_emb.shape[1]
        self.word_num = init_word_emb.shape[0]
        self.word_embedding_layer = nn.Embedding(self.word_num, self.word_emb_size)
        self.word_embedding_layer.from_pretrained(init_word_emb)
        if not emb_train:
            freeze_layer(self.word_embedding_layer)
        self.user_embedding_size = user_emd_size
        self.word_rnn = nn.GRU(input_size=self.word_emb_size, hidden_size=user_emd_size, num_layers=1, batch_first=True)
        self.news_rnn = nn.GRU(input_size=user_emd_size, hidden_size=user_emd_size, num_layers=1, batch_first=True)
        # attention
        self.user_W0 = nn.Linear(user_emd_size, 1)
        self.user_W1 = nn.Linear(user_emd_size, user_emd_size)
        self.user_W2 = nn.Linear(user_emd_size, user_emd_size)

        self.word_W0 = nn.Linear(user_emd_size, 1)
        self.word_W1 = nn.Linear(user_emd_size, user_emd_size)
        self.word_W2 = nn.Linear(user_emd_size, user_emd_size)

    def word_attention(self, rnn_output, history_length, cur_batch_size):
        lt_out = rnn_output[:, -1, :].expand(history_length, cur_batch_size, self.user_embedding_size).permute(1, 0, 2)
        attn_w = self.word_W0(self.word_W1(rnn_output) + self.word_W2(lt_out)).permute(0, 2, 1)
        return F.softmax(attn_w, dim=2)

    def user_attention(self, rnn_output, history_length, cur_batch_size):
        lt_out = rnn_output[:, -1, :].expand(history_length, cur_batch_size, self.user_embedding_size).permute(1, 0, 2)
        attn_w = self.user_W0(self.user_W1(rnn_output) + self.user_W2(lt_out)).permute(0, 2, 1)
        return F.softmax(attn_w, dim=2)

    def forward(self, clicked, candidates):
        # clicked: (batch_size, history_length, title_length)
        # get clicked news tensor
        clicked = self.word_embedding_layer(clicked)
        clicked = F.dropout(clicked, p=0.2)
        clicked_batch_size, history_length, title_length, embedding_size = clicked.shape
        clicked = clicked.reshape(clicked_batch_size*history_length, title_length, embedding_size)
        word_output, h_n = self.word_rnn(clicked, None)
        attn_w = self.word_attention(word_output, title_length, clicked_batch_size*history_length)
        word_output = torch.matmul(attn_w, word_output)
        # word_output = torch.mean(word_output, dim=1)  # average
        word_output = word_output.reshape(clicked_batch_size, history_length, self.user_embedding_size)

        # get user embedding from history click ()
        user_output, h_n = self.news_rnn(word_output, None)  # (16, 35, 100)
        attn_w = self.user_attention(user_output, history_length, clicked_batch_size)
        user_history_embedding = torch.matmul(attn_w, user_output).reshape(clicked_batch_size, self.user_embedding_size)
        # get candidate news tensor
        # user_history_embedding = torch.mean(user_output, dim=1)
        candidates = self.word_embedding_layer(candidates)
        candidates = F.dropout(candidates, p=0.2)
        candidate_batch_size, history_length, title_length, embedding_size = candidates.shape
        candidates = candidates.reshape(candidate_batch_size*history_length, title_length, embedding_size)
        candidate_output, h_n = self.word_rnn(candidates, None)
        attn_w = self.word_attention(candidate_output, title_length, candidate_batch_size*history_length)
        candidate_output = torch.matmul(attn_w, candidate_output)
        # candidate_output = torch.mean(candidate_output, dim=1)

        # (batch_size, history_length, news_embedding_size)
        candidate_output = candidate_output.reshape(candidate_batch_size, history_length, self.user_embedding_size)
        # dot
        tmp = user_history_embedding.unsqueeze(2)
        final_res = torch.matmul(candidate_output, tmp).reshape(candidate_batch_size, -1)
        final_res = F.softmax(final_res, dim=1)
        return final_res.squeeze()


# torch.nn.MultiheadAttention 应该使用的是Narrow self-attention机制，
# 即，把embedding分割成num_heads份，每一份分别拿来做一下attention
class DoubleSelfAttention(nn.Module):
    def __init__(self, init_word_emb, head_num, emb_train):
        super(DoubleSelfAttention, self).__init__()
        self.word_emb_dim = init_word_emb.shape[1]
        self.word_embedding_layer = nn.Embedding(init_word_emb.shape[0], self.word_emb_dim)
        self.word_embedding_layer.from_pretrained(init_word_emb)
        if not emb_train:
            freeze_layer(self.word_embedding_layer)

        self.word_attention = nn.MultiheadAttention(embed_dim=self.word_emb_dim, num_heads=head_num, dropout=0.2)
        self.word_dense = nn.Linear(self.word_emb_dim, 1)

        self.news_attention = nn.MultiheadAttention(embed_dim=self.word_emb_dim, num_heads=head_num, dropout=0.2)
        self.news_dense = nn.Linear(self.word_emb_dim, 1)

    def forward(self, clicked, candidates):
        # clicked: (batch_size, history_length, title_length)
        # change clicked news tensor dimension
        # multi_head input: (title_length, history_length(batch_size), word_embedding_size)
        clicked = self.word_embedding_layer(clicked)
        clicked_batch_size, history_length, title_length, embedding_size = clicked.shape

        attention_input = clicked.reshape(clicked_batch_size*history_length, title_length, embedding_size)
        # (title_length, history_length*batch_size, word_embedding_size)
        attention_input = attention_input.permute(1, 0, 2)
        attn_out, attn_out_weight = self.word_attention(attention_input, attention_input, attention_input)
        attention_weight = self.word_dense(attn_out)
        attention_weight = F.softmax(attention_weight, dim=0)
        batch_title_embedding = torch.matmul(attention_weight.permute(1, 2, 0),
                                             attention_input.permute(1, 0, 2)).permute(1, 0, 2)
        batch_title_embedding = F.tanh(batch_title_embedding)  # (history_length, title_length, word_embedding_size)
        batch_title_embedding = batch_title_embedding.reshape(clicked_batch_size, history_length, embedding_size)

        # (batch_size, history_length, news_embedding_size)
        attention_input = batch_title_embedding.permute(1, 0, 2)
        attention_weight = self.news_dense(self.news_attention(attention_input, attention_input, attention_input)[0])
        attention_weight = F.softmax(attention_weight, dim=0)
        user_history_embedding = torch.matmul(attention_weight.permute(1, 2, 0), attention_input.permute(1, 0, 2))
        # (title_length, history_length(batch_size), word_embedding_size)

        candidates = self.word_embedding_layer(candidates)
        candidate_batch_size, history_length, title_length, embedding_size = candidates.shape
        attention_input = candidates.reshape(candidate_batch_size * history_length, title_length, embedding_size)
        attention_input = attention_input.permute(1, 0, 2)
        attention_weight = self.word_dense(self.word_attention(attention_input, attention_input, attention_input)[0])
        attention_weight = F.softmax(attention_weight, dim=0)
        batch_title_embedding = torch.matmul(attention_weight.permute(1, 2, 0),
                                             attention_input.permute(1, 0, 2)).permute(1, 0, 2)
        # (history_length, title_length, word_embedding_size)
        batch_title_embedding = batch_title_embedding.reshape(candidate_batch_size, history_length, embedding_size)

        # dot
        user_history_embedding = user_history_embedding.permute(0, 2, 1)
        final_res = torch.matmul(batch_title_embedding, user_history_embedding).reshape(candidate_batch_size, -1)
        final_res = F.softmax(final_res, dim=1)
        return final_res.squeeze()  # (64, 5)


class DoubleCNN(nn.Module):
    def __init__(self, init_word_emb, wd_out_chn, us_out_chn, emb_train):
        super(DoubleCNN, self).__init__()
        self.word_emb_size = init_word_emb.shape[1]
        self.word_num = init_word_emb.shape[0]

        self.word_embedding_layer = nn.Embedding(self.word_num, self.word_emb_size)
        self.word_embedding_layer.from_pretrained(init_word_emb)
        if not emb_train:
            freeze_layer(self.word_embedding_layer)
        self.word_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=wd_out_chn,
                kernel_size=(15, 1)
            ),
            nn.ReLU()
        )
        self.word_flat = nn.Linear(wd_out_chn, 1)

        self.news_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=us_out_chn,
                kernel_size=(35, 1)
            ),
            nn.ReLU()
        )
        self.user_flat = nn.Linear(us_out_chn, 1)

    def forward(self, clicked, candidates):
        clicked = self.word_embedding_layer(clicked)
        clicked = F.dropout(clicked, p=0.2)
        clicked_batch_size, history_length, title_length, embedding_size = clicked.shape
        clicked = clicked.reshape(clicked_batch_size * history_length, -1, title_length, embedding_size)
        cli_title_emb = self.word_conv(clicked).permute(0, 3, 2, 1)
        cli_title_emb = self.word_flat(cli_title_emb).reshape(clicked_batch_size, -1, history_length, embedding_size)

        user_emb = self.news_conv(cli_title_emb).permute(0, 3, 2, 1)
        user_emb = self.user_flat(user_emb).reshape(clicked_batch_size, embedding_size, -1)

        candidates = self.word_embedding_layer(candidates)
        candidates = F.dropout(candidates, p=0.2)
        candidate_batch_size, history_length, title_length, embedding_size = candidates.shape
        candidates = candidates.reshape(candidate_batch_size * history_length, -1, title_length, embedding_size)
        candi_title_emb = self.word_conv(candidates).permute(0, 3, 2, 1)
        candi_title_emb = self.word_flat(candi_title_emb).reshape(candidate_batch_size, history_length, embedding_size)

        final_res = torch.matmul(candi_title_emb, user_emb).reshape(candidate_batch_size, -1)
        final_res = F.softmax(final_res, dim=1)
        return final_res.squeeze()



