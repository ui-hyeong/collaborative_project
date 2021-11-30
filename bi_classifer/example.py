# -*- coding: utf-8 -*-
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel

from Cooperation_project.Builder import Build_X
import re
import emoji
from soynlp.normalizer import repeat_normalize
from torch.nn import functional as F

from bi_classifer.SimpleDataset import SimpleDataset

lyrics = pd.read_csv(r'C:\Users\jeonguihyeong\PycharmProjects\Cooperation_project\bi_classifer\DEC_cluster.csv')
DATA = lyrics.values.tolist()

#
# emojis = list({y for x in emoji.UNICODE_EMOJI.values() for y in x.keys()})
# emojis = ''.join(emojis)
# pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
# url_pattern = re.compile(
#     r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
#
# def clean(x):
#     x = pattern.sub(' ', x)
#     x = url_pattern.sub('', x)
#     x = x.strip()
#     x = repeat_normalize(x, num_repeats=2)
#     return x

# tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")
#
# bertmodel = AutoModelForMaskedLM.from_pretrained("monologg/kobigbird-bert-base")
#
# tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
#
# bertmodel = AutoModelForMaskedLM.from_pretrained("beomi/kcbert-base")

tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
bertmodel = AutoModel.from_pretrained("monologg/kobert")
sents = [sent for _, sent, _, _, _ in DATA]
labels = [label for _, _, label, _, _ in DATA]
# data = clean(data)
print(data)
device = torch.device('cuda')
X = Build_X(data, tokenizer, device=device)
dataset = SimpleDataset(sents_test)
# input_ids = X[:, 0]
# token_type_ids = X[:, 1]
# attention_mask = X[:, 2]
# H_all= bertmodel(input_ids, token_type_ids, attention_mask)
# # [:, 0, :]
#
# # print(H_all[1].shape)
# print(H_all[0].shape)
# print(pooler)
# print(bertmodel.config)

model = torch.load('model.pth', map_location=device)
x = model.predict(X)
print(x)
x = F.softmax(x, dim=1)
print(x)