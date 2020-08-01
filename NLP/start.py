# -*- coding: utf-8 -*-
# @File: start.py
# @Author: Pei Yu
# @Date:   2020-07-28

import torch
import torch.nn as nn
import pickle
import numpy as np
import warnings
import tqdm
import os
from sklearn.metrics import roc_auc_score
import random
from model import DoubleGRU, DoubleSelfAttention, DoubleCNN

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3, 4"
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    test_num = 5000
    validate_num = 1000
    batch_size = 64
    epoch_num = 100
    random.seed(1)

    word_embedding_size = 100
    user_embedding_size = 50
    history_length_ = 35
    title_length_ = 15
    emb_train_ = True
    
    heads_num = 10

    user_out_channel = 32
    word_out_channel = 32

    # load train data, (tensor, tensor, tensor, list)
    init_word_embedding, train_clicked_news, train_candidate_news, train_label = \
        pickle.load(open("train_available_data_" + str(word_embedding_size) + "d_3d.txt", "rb"))
    init_word_embedding = init_word_embedding.cuda()
    train_clicked_news = train_clicked_news.cuda()
    train_candidate_news = train_candidate_news.cuda()
    n_batch = len(train_clicked_news) // batch_size
    random_index = np.array(random.sample(list(range(train_clicked_news.shape[0])), k=validate_num))

    # load test data for calculating auc
    word_embedding, test_clicked_news, test_candidate_news, test_label \
        = pickle.load(open("test_available_data_" + str(word_embedding_size) + "d_3d.txt", "rb"))
    test_clicked_news, test_candidate_news, test_label = \
        test_clicked_news[:test_num], test_candidate_news[:test_num], test_label[:test_num]

    # load validate data for calculating acc
    validate_clicked_news, validate_candidate_news, validate_label = \
        train_clicked_news[random_index], train_candidate_news[random_index], np.array(train_label)[random_index]

    # model
    model_type = input()
    if model_type == "GRU":
        model = DoubleGRU(user_embedding_size, init_word_embedding, emb_train_)
        print("model: {}, word embedding size: {}, user embedding size: {}, embedding train: {}".
              format(model_type, word_embedding_size, user_embedding_size, emb_train_))
    elif model_type == "CNN":
        model = DoubleCNN(init_word_embedding, us_out_chn=32, wd_out_chn=32, emb_train=emb_train_)
        print("model: {}, word embedding size: {}, user out channel: {}, word out channel: {}, embedding train: {}".
              format(model_type, word_embedding_size, user_out_channel, word_out_channel, emb_train_))
    elif model_type == "self-attention":
        model = DoubleSelfAttention(init_word_embedding, heads_num, emb_train_)
        print("word embedding size: {}, head num: {}, embedding train: {}".format(word_embedding_size,
                                                                                  heads_num,
                                                                                  emb_train_))
    word_mask = torch.tril(torch.ones(title_length_, title_length_)).cuda()
    news_mask = torch.tril(torch.ones(history_length_, history_length_)).cuda()
    # model = torch.load("results/trained_model.pkl")
    model = nn.DataParallel(model).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss().cuda()

    total_loss = []
    total_auc = []
    total_acc = []
    for epoch in range(epoch_num):
        batch_loss = []
        batch_acc = []

        for i in tqdm.tqdm(range(n_batch)):
            loss = 0
            optimizer.zero_grad()

            # generating batch data and padding
            batch_train_label = np.argmax(train_label[i*batch_size:(i+1)*batch_size], axis=1)
            batch_train_clicked_news = train_clicked_news[i*batch_size:(i+1)*batch_size]
            batch_train_candidate_news = train_candidate_news[i*batch_size:(i+1)*batch_size]

            # training
            output = model(batch_train_clicked_news, batch_train_candidate_news)
            loss += loss_func(output, torch.Tensor(batch_train_label).long().cuda())

            # calculating loss and backward
            batch_loss.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
            # print("\n -->loss: {}".format(loss))  # loss on each batch

            # for name, parms in model.named_parameters():
            #     print('-->name:', name, '-->grad_requires:', parms.requires_grad, '-->weight', parms.data,
            #           '-->grad_value:', parms.grad)
        # testing each epoch
        print("epoch : {}".format(epoch))
        avg_auc = []
        for tn in range(len(test_clicked_news)):
            prediction = model(test_clicked_news[tn].unsqueeze(0).cuda(),
                               torch.Tensor(test_candidate_news[tn]).long().unsqueeze(0).cuda()).cpu().detach().numpy()
            avg_auc.append(roc_auc_score(test_label[tn], prediction))
        print(" -->auc: {}".format(sum(avg_auc) / len(avg_auc)))  # auc on each batch

        # validating
        prediction = model(validate_clicked_news, validate_candidate_news)

        acc = sum(np.argmax(prediction.cpu().detach().numpy(), axis=1) ==
                  np.argmax(validate_label, axis=1)) / validate_num
        batch_acc.append(acc)
        print(" -->acc: {}".format(acc))  # acc on each batch

        # recoding
        total_acc.append(batch_acc)
        total_loss.append(batch_loss)
        total_auc.append(avg_auc)
        pickle.dump((batch_loss, batch_acc, avg_auc), open("results/loss_acc_auc_epoch"+str(epoch)+".txt", "wb"))
    pickle.dump((total_loss, total_acc, total_auc), open("results/loss_acc_auc_total.txt", "wb"))
    torch.save(model, "results/trained_model.pkl")
