# -*- coding: utf-8 -*-
# @File: data_process.py
# @Author: Pei Yu
# @Date:   2020-07-28

import pickle
from nltk.tokenize import word_tokenize
import numpy as np
from ast import literal_eval
import pandas as pd
import tqdm
import torch


def generate_sample(clicked, candidate, label):
    """
    generate np_ratio negative samples per one positive samples.
    :param clicked: shape:(history_length, title_length, word_embedding_size)
    :param candidate: shape:(impressions_length, title_length, word_embedding_size)
    :param label: shape:(impressions_length, )
    :return: positive_num * clicked, positive_num * candidate, positive_num * [np_ratio*0, 1*1]
    """
    np_ratio = 4
    positive_index = [i for i in range(len(label)) if label[i] == 1]
    negative_index = [j for j in range(len(label)) if label[j] == 0]
    total_clicked_news_ = []
    total_candidate_news_ = []
    total_label_ = []
    for pos_index in positive_index:
        total_clicked_news_.append(clicked.copy())
        label_ = np.random.permutation([1] + [0 for i in range(np_ratio)])
        total_label_.append(label_)
        batch_candidate = []
        for n in range(np_ratio):
            batch_candidate.append(candidate[np.random.choice(negative_index)])
        batch_candidate.insert(np.argmax(label_), candidate[pos_index])
        total_candidate_news_.append(batch_candidate)
    return total_clicked_news_, total_candidate_news_, total_label_


def text_process():
    """
    load all news both the train data and the test data, generate the dict of news and word.
    :return: news: {"news_id": {"title": [word_index, ...], "entity": [...]}}, word_dict: {"word": word_index}
    """
    # load stop words
    print("loading stop words...")
    stop_words = set()
    with open("stop_words.txt", encoding="utf-8") as f:
        for line in tqdm.tqdm(f.readlines()):
            stop_words.add(line.strip('\n'))

    # generate news information dict {news_id:{"title": [...], "entities":[....]}}
    print("generating news information dict...")
    news = {}
    word_dict = {"PADDING": 0}  # word to index
    for file_name in ["train", "test"]:
        news_info = pd.read_csv(file_name+"/news.tsv",
                                sep='\t',
                                names=["news", "category", "subcategory", "title",
                                       "abstract", "URL", "title_entities","abstract_entities"])
        for index, row in tqdm.tqdm(news_info.iterrows()):
            try:
                if row["news"] not in news:
                    news[row["news"]] = {}
                news[row["news"]]["title"] = []
                news[row["news"]]["entities"] = []

                for word in word_tokenize(row["title"]):
                    word = word.lower()
                    if word not in stop_words:
                        if word not in word_dict:
                            word_dict[word] = len(word_dict)
                        news[row["news"]]["title"].append(word_dict[word])

                news[row["news"]]["title"] = np.array(news[row["news"]]["title"])  # to np.array

                to_list = literal_eval(row["title_entities"])
                for entity in to_list:
                    # TODO: insert entity embedding
                    news[row["news"]]["entities"].append(entity["WikidataId"])
            except ValueError:
                continue
    return news, word_dict


def candidate_structure_data(raw_candidate_news, title_length):
    """
    structure the candidate news for the input of net, fixed the title length to title_length.
    :param raw_candidate_news: (batch_size, history_length, title_length)
    :param title_length:
    :return: structure_candidate_news
    """
    structure_candidate_news = []
    for candidate_news in raw_candidate_news:
        structure_title = []
        for title in candidate_news:
            if title_length - title.shape[0] > 0:
                word_padding = np.zeros(shape=[title_length - title.shape[0]])
                structure_title.append(np.concatenate((title, word_padding), axis=0))
            else:
                structure_title.append(title[:title_length])
        structure_title = np.array(structure_title)
        structure_candidate_news.append(structure_title)
    return structure_candidate_news


def clicked_structure_data(raw_clicked_news, history_length, title_length):
    """
    structure the candidate news for the input of net, fixed the title length to title_length,
    and fixed the history length to history_length.
    :param raw_clicked_news: (batch_size, history_length, title_length)
    :param history_length:
    :param title_length:
    :return: structure_clicked_news
    """
    structure_clicked_news = []
    for clicked_news in raw_clicked_news:
        structure_title = []
        for title in clicked_news:
            if title_length - title.shape[0] > 0:
                word_padding = np.zeros(title_length-title.shape[0])
                structure_title.append(np.concatenate((title, word_padding)))
            else:
                structure_title.append(title[:title_length])

        structure_title = np.array(structure_title)
        if history_length - structure_title.shape[0] > 0:
            history_padding = np.zeros(shape=[history_length-structure_title.shape[0], title_length])
            structure_clicked_news.append(np.concatenate((structure_title, history_padding), axis=0))
        else:
            structure_clicked_news.append(structure_title[:history_length])
    return structure_clicked_news


def generate_data(news, train, history_length, title_length):
    """
    generate the net training data.
    :param news: news_dict
    :param train: train or test?
    :param history_length:
    :param title_length:
    :return: total_clicked_news, total_candidate_news, total_label
    """
    # load train behaviors
    if train:
        file_name = "train"
    else:
        file_name = "test"
    print("loading " + file_name + " behaviors...")
    total_clicked_news = []
    total_candidate_news = []
    total_label = []
    behaviors = pd.read_csv(file_name + "/behaviors.tsv", sep='\t', names=["user", "time", "history", "impressions"])
    for index, row in tqdm.tqdm(behaviors.iterrows()):
        clicked_news = []
        candidate_news = []
        label = []
        try:
            # get user history
            for news_id in row["history"].split():
                if len(news[news_id]["title"]) != 0:
                    clicked_news.append(news[news_id]["title"])
            # get user impressions
            for news_id in row["impressions"].split():
                candidate_news.append(news[news_id[:-2]]["title"])
                label.append(eval(news_id[-1]))

            # clicked_news, candidate_news, label
            # generate samples(four negative samples, one positive sample)
            if train:
                clicked_news_list, candidate_news_list, label_list = generate_sample(clicked_news, candidate_news, label)
                if len(clicked_news) != 0 and len(candidate_news) != 0:
                    for element_clicked in clicked_news_list:
                        total_clicked_news.append(np.array(element_clicked))
                    for element_candi in candidate_news_list:
                        total_candidate_news.append(np.array(element_candi))
                    for element_label in label_list:
                        total_label.append(np.array(element_label))
            else:
                if len(clicked_news) != 0 and len(candidate_news) != 0:
                    total_clicked_news.append(np.array(clicked_news))
                    total_candidate_news.append(np.array(candidate_news))
                    total_label.append(np.array(label))
        except AttributeError:
            continue
    print("structure " + file_name + " data...")

    # !!!!! the length of candidate news in test case is not the same, so can't convert to the tensor !!!!!
    total_clicked_news = torch.Tensor(clicked_structure_data(total_clicked_news, history_length, title_length)).long()
    if train:
        total_candidate_news = torch.Tensor(candidate_structure_data(total_candidate_news, title_length)).long()
    else:
        total_candidate_news = candidate_structure_data(total_candidate_news, title_length)

    # train: (tensor, tensor, list), test: (tensor, list, list)
    return total_clicked_news, total_candidate_news, total_label


def generate_word_embedding(word_dict, word_vec_file, word_embedding_size):  # "glove.6B/glove.6B.300d.txt"
    """
    generate word embedding.
    :param word_dict:
    :param word_vec_file:
    :param word_embedding_size:
    :return: embedding_matrix: torch.Tensor
    """
    # load word embedding
    print("loading word embedding...")
    embedding_matrix = []
    word_embedding = {}
    with open(word_vec_file, encoding="utf-8") as f:
        for line in tqdm.tqdm(f.readlines()):
            cur = []
            for value in line.split()[1:]:
                cur.append(eval(value))
            word_embedding[line.split()[0]] = np.array(cur)
            assert word_embedding_size == len(cur)
    print("generating embedding matrix...")
    for word in word_dict:
        if word not in word_embedding:
            random_init = np.random.randn(word_embedding_size)
            scaled_random_init = (random_init - random_init.min()) * 2 / (random_init.max() - random_init.min()) - 1
            embedding_matrix.append(scaled_random_init)
        else:
            embedding_matrix.append(word_embedding[word])
    return torch.Tensor(embedding_matrix)


def save_data(embedding_matrix, clicked_news, candidate_news, total_label, flag, wd):
    """
    save data.
    :param embedding_matrix:
    :param clicked_news:
    :param candidate_news:
    :param total_label:
    :param flag: train or test?
    :return: file
    """
    print("saving data...")
    if flag:
        file_name = "train"
    else:
        file_name = "test"
    pickle.dump((embedding_matrix, clicked_news, candidate_news, total_label),
                open(file_name + "_available_data_" + str(wd) + "d_3d.txt", "wb"))


if __name__ == "__main__":
    file = ["test", "train"]
    history_length_ = 35
    title_length_ = 15
    word_embedding_size_ = 300

    word_vec_file_ = "glove.6B/glove.6B." + str(word_embedding_size_) + "d.txt"

    news_, word_dict_ = text_process()
    embedding_matrix_ = generate_word_embedding(word_dict_, word_vec_file_, word_embedding_size_)
    for train_ in [1, 0]:
        print("===============>> processing " + file[train_] + " data <<===============")
        total_clicked_news_, total_candidate_news_, total_label_ = generate_data(news_,
                                                                                 train_,
                                                                                 history_length_,
                                                                                 title_length_)
        # tensor, tensor, tensor, list
        save_data(embedding_matrix_, total_clicked_news_, total_candidate_news_, total_label_, train_,
                  word_embedding_size_)
