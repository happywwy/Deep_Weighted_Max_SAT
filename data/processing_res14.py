import _pickle as cPickle
import numpy as np
import re
import torch
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/Desktop/DeepWMaxSAT/')


args = parser.parse_args()
opt = vars(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

uni_token_pos_res_train = cPickle.load(open(opt['path'] + "data/Aspect_Opinion/uni_token_pos_res14", "rb"))
uni_token_pos_res_test = cPickle.load(open(opt['path'] + "data/Aspect_Opinion/uni_token_pos_restest14", "rb"))

f_aspect_train = open(opt['path'] + "data/Aspect_Opinion/aspectTerm_res14", "r")
f_aspect_test = open(opt['path'] + "data/Aspect_Opinion/aspectTerm_restest14", "r")

f_opinion_train = open(opt['path'] + "data/Aspect_Opinion/sentence_res14_op", "r")
f_opinion_test = open(opt['path'] + "data/Aspect_Opinion/sentence_restest14_op", "r")

words_train = [[item[0] for item in uni] for uni in uni_token_pos_res_train]
words_test = [[item[0] for item in uni] for uni in uni_token_pos_res_test]

pos_train = [[item[1] for item in uni] for uni in uni_token_pos_res_train]
pos_test = [[item[1] for item in uni] for uni in uni_token_pos_res_test]

aspects_train = f_aspect_train.read().splitlines()
aspects_test = f_aspect_test.read().splitlines()
sentences_op_train = f_opinion_train.read().splitlines()
sentences_op_test = f_opinion_test.read().splitlines()

f_aspect_train.close()
f_aspect_test.close()
f_opinion_train.close()
f_opinion_test.close()

dic_file = open(opt['path'] + "data/Aspect_Opinion/w2v_yelp300_10.txt", "r")
dic = dic_file.readlines()
dic_file.close()

f_text_train = open(opt['path'] + "data/Aspect_Opinion/sentence_res14", "r")
f_text_test = open(opt['path'] + "data/Aspect_Opinion/sentence_restest14", "r")
sentences_train = f_text_train.read().splitlines()
sentences_test = f_text_test.read().splitlines()
f_text_train.close()
f_text_test.close()

f_raw_parses_res_train = open(opt['path'] + 'data/Aspect_Opinion/raw_parses_res14')
f_raw_parses_res_test = open(opt['path'] + 'data/Aspect_Opinion/raw_parses_restest14')

parse_train = f_raw_parses_res_train.read().splitlines()
parse_test = f_raw_parses_res_test.read().splitlines()

f_raw_parses_res_train.close()
f_raw_parses_res_test.close()

labels_train = []
labels_test = []

#Generate training labels
n = 0
for sentence_op, aspect in zip(sentences_op_train, aspects_train):
    tokens = words_train[n]
    label = [0 for i in range(len(tokens))]
    if "##" in sentence_op:
        opinion = sentence_op.split("##")[1]
        # for res16 data
        # opinion_list = opinion.split(",")[:-1]
        # opinion_set = [item.split(" - ")[0] for item in opinion_list]
        # for laptop, res14
        opinion_list = opinion.split(", ")
        opinion_set = [' '.join(item.split(" ")[:-1]) for item in opinion_list]
        for op in opinion_set:
            op = op.strip()
            if " " not in op:
                for i, tok in enumerate(tokens):
                    if tok.lower() == op.lower():
                        label[i] = 3
            else:
                op = op.split()

                for i, tok in enumerate(tokens):
                    if tok.lower() == op[0].lower() and len(tokens) >= i + len(op):
                        match = True
                        for j in range(1, len(op)):
                            if tokens[i + j].lower() != op[j].lower():
                                match = False
                                break
                    else:
                        match = False
                    if match:
                        label[i] = 3
                        for j in range(1, len(op)):
                            label[i + j] = 4

    if "NIL" not in aspect:
        aspect_list = aspect.split(",")
        for asp in aspect_list:
            asp = asp.strip()
            if " " not in asp:
                for i, tok in enumerate(tokens):
                    if tok == asp:
                        label[i] = 1
            else:
                asp = asp.split()

                for i, tok in enumerate(tokens):
                    if tok == asp[0] and len(tokens) >= i + len(asp):
                        match = True
                        for j in range(1, len(asp)):
                            if tokens[i + j] != asp[j]:
                                match = False
                                break
                    else:
                        match = False
                    if match:
                        label[i] = 1
                        for j in range(1, len(asp)):
                            label[i + j] = 2

    labels_train.append(label)
    print(n)
    n += 1

#Generate testing labels
n = 0
for sentence_op, aspect in zip(sentences_op_test, aspects_test):
    #print(aspect)
    tokens = words_test[n]
    label = [0 for i in range(len(tokens))]
    if "##" in sentence_op:
        opinion = sentence_op.split("##")[1]
        # for res16 data
        # opinion_list = opinion.split(",")[:-1]
        # opinion_set = [item.split(" - ")[0] for item in opinion_list]
        # for laptop, res14
        opinion_list = opinion.split(", ")
        opinion_set = [' '.join(item.split(" ")[:-1]) for item in opinion_list]
        for op in opinion_set:
            op = op.strip()
            if " " not in op:
                for i, tok in enumerate(tokens):
                    if tok.lower() == op.lower():
                        label[i] = 3
            else:
                op = op.split()

                for i, tok in enumerate(tokens):
                    if tok.lower() == op[0].lower() and len(tokens) >= i + len(op):
                        match = True
                        for j in range(1, len(op)):
                            if tokens[i + j].lower() != op[j].lower():
                                match = False
                                break
                    else:
                        match = False
                    if match:
                        label[i] = 3
                        for j in range(1, len(op)):
                            label[i + j] = 4

    if "NIL" not in aspect:
        aspect_list = aspect.split(",")
        for asp in aspect_list:
            asp = asp.strip()
            if " " not in asp:
                for i, tok in enumerate(tokens):
                    if tok == asp:
                        label[i] = 1
            else:
                asp = asp.split()

                for i, tok in enumerate(tokens):
                    if tok == asp[0] and len(tokens) >= i + len(asp):
                        match = True
                        for j in range(1, len(asp)):
                            if tokens[i + j] != asp[j]:
                                match = False
                                break
                    else:
                        match = False
                    if match:
                        label[i] = 1
                        for j in range(1, len(asp)):
                            label[i + j] = 2

    labels_test.append(label)
    print(n)
    n += 1


dictionary = {}
count = 0

for line in dic:
    word_vector = line.split(",")[:-1]
    vector_list = []
    for element in word_vector[len(word_vector) - 300:]:
        vector_list.append(float(element))
    word = ','.join(word_vector[:len(word_vector) - 300])

    vector = np.asarray(vector_list)
    dictionary[word] = vector

idxs_train = []
idxs_test = []
vocab = ["ppaadd", "punkt", "unk"]
e_pad = np.asarray([(2 * np.random.rand() - 1) for i in range(300)])
e_unk = np.asarray([(2 * np.random.rand() - 1) for i in range(300)])
e_punkt = np.asarray([(2 * np.random.rand() - 1) for i in range(300)])
embedding = [e_pad, e_punkt, e_unk]

#Generate training idxs
n = 0
for sentence in sentences_train:
    tokens = words_train[n]
    idx = []
    for token in tokens:
        token = token.lower()
        if token.isalpha():
            if token not in vocab:
                if token in dictionary.keys():
                    embedding.append(dictionary[token])
                    vocab.append(token)
                    idx.append(vocab.index(token))
                else:
                    count += 1
                    embedding.append(np.asarray([(2 * np.random.rand() - 1) for i in range(300)]))
                    vocab.append(token)
                    idx.append(vocab.index(token))
            else:
                idx.append(vocab.index(token))

        else:
            idx.append(vocab.index("punkt"))

    idxs_train.append(idx)
    n += 1

#Generate testing idxs
n = 0
for sentence in sentences_test:
    tokens = words_test[n]
    idx = []
    for token in tokens:
        token = token.lower()
        if token.isalpha():
            if token not in vocab:
                if token in dictionary.keys():
                    embedding.append(dictionary[token])
                    vocab.append(token)
                    idx.append(vocab.index(token))
                else:
                    count += 1
                    idx.append(vocab.index("unk"))
            else:
                idx.append(vocab.index(token))

        else:
            idx.append(vocab.index("punkt"))

    idxs_test.append(idx)
    n += 1

embedding = np.asarray(embedding)

parse_pos_train = [[]for i in range(len(idxs_train))]
i = 0
for k in range(len(parse_train)):
    txt = parse_train[k].replace('-', ',')
    txt = re.split(r'[(,)]', txt)
    if txt != ['']:
        word_1, word_2 = [x.strip() for x in txt if x.strip().isnumeric()]
        if word_1 != '0' and word_2 != '0':
                #word_1_idx, word_2_idx = idxs_train[i][int(word_1)-1], idxs_train[i][int(word_2)-1]
                word_1_pos, word_2_pos = pos_train[i][int(word_1) - 1], pos_train[i][int(word_2) - 1]
                print(i, k, [word_1, word_1_pos, word_2, word_2_pos, txt[0]])
                #parse_pos_train[i].append([word_1_idx, word_1_pos, word_2_idx, word_2_pos, txt[0]])
                parse_pos_train[i].append([int(word_1) - 1, word_1_pos, int(word_2) - 1, word_2_pos, txt[0]])
    else:
        i += 1

parse_pos_test = [[]for i in range(len(idxs_test))]
i = 0
for k in range(len(parse_test)):
    txt = parse_test[k].replace('-', ',')
    txt = re.split(r'[(,)]', txt)
    if txt != ['']:
        word_1, word_2 = [x.strip() for x in txt if x.strip().isnumeric()]
        if word_1 != '0' and word_2 != '0':
                #word_1_idx, word_2_idx = idxs_test[i][int(word_1)-1], idxs_test[i][int(word_2)-1]
                word_1_pos, word_2_pos = pos_test[i][int(word_1) - 1], pos_test[i][int(word_2) - 1]
                #print(i, k, [word_1_idx, word_1_pos, word_2_idx, word_2_pos, txt[0]])
                #parse_pos_test[i].append([word_1_idx, word_1_pos, word_2_idx, word_2_pos, txt[0]])
                parse_pos_test[i].append([int(word_1) - 1, word_1_pos, int(word_2) - 1, word_2_pos, txt[0]])
    else:
        i += 1

cPickle.dump((pos_train, pos_test), open(opt['path'] + "data/Aspect_Opinion/pos_res14.tag", "wb"))
cPickle.dump(labels_train, open(opt['path'] + "data/Aspect_Opinion/labels_res14.train", "wb"))
cPickle.dump(labels_test, open(opt['path'] + "data/Aspect_Opinion/labels_res14.test", "wb"))
cPickle.dump(words_train, open(opt['path'] + "data/Aspect_Opinion/words_res14.train", "wb"))
cPickle.dump(words_test, open(opt['path'] + "data/Aspect_Opinion/words_res14.test", "wb"))
cPickle.dump(embedding, open(opt['path'] + "data/Aspect_Opinion/embedding300_res14", "wb"))
cPickle.dump(idxs_train, open(opt['path'] + "data/Aspect_Opinion/idx_res14.train", "wb"))
cPickle.dump(idxs_test, open(opt['path'] + "data/Aspect_Opinion/idx_res14.test", "wb"))
cPickle.dump(parse_pos_train, open(opt['path'] + "data/Aspect_Opinion/parse_pos_res14.train", "wb"))
cPickle.dump(parse_pos_test, open(opt['path'] + "data/Aspect_Opinion/parse_pos_res14.test", "wb"))

