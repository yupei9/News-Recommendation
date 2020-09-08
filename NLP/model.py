# -*- coding: utf-8 -*-
# @File: model.py
# @Author: Pei Yu
# @Date:   2020-07-28

import torch
import torch.nn as nn
from torch.nn import functional as F


class DoubleGRU(nn.Module):
    def __init__(self, user_emb_size,
             news_emb_size,
             init_word_emb,
             word_emb_size,
             sc_num,
             sc_emb,
             bidirectional):
        super(DoubleGRUC1, self).__init__()
        self.word_emb_size = word_emb_size
        self.word_num = init_word_emb.shape[0]

        self.word_embedding_layer = nn.Embedding(self.word_num, self.word_emb_size, padding_idx=0)
        self.word_embedding_layer.weight.data.copy_(init_word_emb)
        with torch.no_grad():
            self.word_embedding_layer.weight[0].fill_(0)
        self.sub_embedding_layer = nn.Embedding(sc_num, sc_emb, padding_idx=0)

        self.user_emb_size = user_emb_size
        self.news_emb_size = news_emb_size
        self.sc_emb_size = sc_emb
        self.bidirectional = bidirectional
        self.layer_size = 2 if self.bidirectional else 1

        self.word_rnn = nn.GRU(input_size=self.word_emb_size,
                               hidden_size=self.news_emb_size,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=self.bidirectional)

        self.input_size = self.news_emb_size*self.layer_size + self.sc_emb_size
        self.news_rnn = nn.GRU(input_size=self.input_size,
                               hidden_size=self.user_emb_size,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=self.bidirectional)

        self.word_W0 = nn.Linear(self.news_emb_size*self.layer_size, 1)
        self.word_W1 = nn.Linear(self.news_emb_size*self.layer_size, self.news_emb_size*self.layer_size)
        self.word_W2 = nn.Linear(self.news_emb_size*self.layer_size, self.news_emb_size*self.layer_size)
        self.word_W3 = nn.Linear(self.news_emb_size*self.layer_size, self.news_emb_size*self.layer_size)
        self.candi_l1 = nn.Linear(self.input_size, self.user_emb_size*self.layer_size)
        # attention
        self.user_W0 = nn.Linear(self.user_emb_size*self.layer_size, 1)
        self.user_W1 = nn.Linear(self.user_emb_size*self.layer_size, self.user_emb_size*self.layer_size)
        self.user_W2 = nn.Linear(self.user_emb_size*self.layer_size, self.user_emb_size*self.layer_size)
        self.user_W3 = nn.Linear(self.user_emb_size * self.layer_size, self.user_emb_size * self.layer_size)


    def word_attention(self, rnn_output, history_length, cur_batch_size):
        lt_out = rnn_output[:, -1, :].contiguous()
        mean_out = torch.mean(rnn_output, dim=1)
        lt_out = lt_out.expand(history_length, cur_batch_size, self.news_emb_size*self.layer_size).permute(1, 0, 2)
        mean_out = mean_out.expand(history_length, cur_batch_size, self.news_emb_size*self.layer_size).permute(1, 0, 2)
        attn_w = self.word_W0(F.leaky_relu(self.word_W1(rnn_output) +
                                           self.word_W2(lt_out) + self.word_W3(mean_out))).permute(0, 2, 1)
        # attn_w = self.word_W0(F.sigmoid(self.word_W1(rnn_output) + self.word_W3(mean_out))).permute(0, 2, 1)

        return F.softmax(attn_w, dim=2)

    def user_attention(self, rnn_output, history_length, cur_batch_size):
        lt_out = rnn_output[:, -1, :].contiguous()
        mean_out = torch.mean(rnn_output, dim=1)
        lt_out = lt_out.expand(history_length, cur_batch_size, self.user_emb_size*self.layer_size).permute(1, 0, 2)
        mean_out = mean_out.expand(history_length, cur_batch_size, self.user_emb_size * self.layer_size).permute(1, 0, 2)
        attn_w = self.user_W0(F.sigmoid(self.user_W1(rnn_output) +
                                        self.user_W2(lt_out) + self.user_W3(mean_out))).permute(0, 2, 1)
        return F.softmax(attn_w, dim=2)

    def forward(self, clicked, clicked_sub, candidates, candi_sub):
        # clicked: (batch_size, history_length, title_length)
        # get clicked news tensor
        clicked = self.word_embedding_layer(clicked)
        clicked = F.dropout(clicked, p=0.2)
        clicked_batch_size, history_length, title_length, embedding_size = clicked.shape
        clicked = clicked.reshape(clicked_batch_size*history_length, title_length, embedding_size)
        word_output, h_n = self.word_rnn(clicked, None)
        attn_w = self.word_attention(word_output, title_length, clicked_batch_size*history_length)
        word_output = torch.matmul(attn_w, word_output)
        word_output = word_output.reshape(clicked_batch_size, history_length, self.news_emb_size*self.layer_size)

        clicked_sub = clicked_sub[:, :, 0]
        sc_embed = self.sub_embedding_layer(clicked_sub)
        word_output = torch.cat((word_output, sc_embed), dim=2)

        # get user embedding from history click ()
        user_output, h_n = self.news_rnn(word_output, None)  # (16, 35, 100)
        attn_w = self.user_attention(user_output, history_length, clicked_batch_size)
        user_history_embedding = torch.matmul(attn_w, user_output).reshape(clicked_batch_size,
                                                                           self.user_emb_size*self.layer_size)
        # get candidate news tensor
        # user_history_embedding = torch.mean(user_output, dim=1)
        candidates = self.word_embedding_layer(candidates)
        candidates = F.dropout(candidates, p=0.2)
        candidate_batch_size, history_length, title_length, embedding_size = candidates.shape
        candidates = candidates.reshape(candidate_batch_size*history_length, title_length, embedding_size)
        candidate_output, h_n = self.word_rnn(candidates, None)
        attn_w = self.word_attention(candidate_output, title_length, candidate_batch_size*history_length)
        candidate_output = torch.matmul(attn_w, candidate_output)

        # (batch_size, history_length, news_embedding_size)
        candidate_output = candidate_output.reshape(candidate_batch_size,
                                                    history_length,
                                                    self.news_emb_size*self.layer_size)
        candi_sub = candi_sub[:, :, 0]
        cs_embed = self.sub_embedding_layer(candi_sub)
        candidate_output = F.sigmoid(self.candi_l1(torch.cat((candidate_output, cs_embed), dim=2)))
        # candidate (batch_size, candi_num, user_emb_size)
        tmp = user_history_embedding.unsqueeze(2)  # (batch_size, user_emb_size, 1)
        final_res = torch.matmul(candidate_output, tmp).reshape(candidate_batch_size, -1)
        # final_res = F.softmax(final_res, dim=1)
        return final_res.squeeze()


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



