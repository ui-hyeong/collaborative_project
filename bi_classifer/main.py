from typing import List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim import SGD
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from torch.utils.data import DataLoader


from transformers.optimization import get_cosine_schedule_with_warmup, AdamW

from bi_classifer.Broken_up_Classifer import Broken_up_Classifer
from bi_classifer.Builder import Build_X, Build_y
from bi_classifer.SimpleDataset import SimpleDataset
from bi_classifer.train_test import train_test

lyrics = pd.read_csv(r'C:\Users\jeonguihyeong\PycharmProjects\Cooperation_project\bi_classifer\DEC_cluster.csv')
# ox_0 = lyrics[(lyrics['긍부정'] == 0) & (lyrics['이별유무'] == 1)]  # 이별, 부정
# ox_1 = lyrics[(lyrics['긍부정'] == 1) & (lyrics['이별유무'] == 1)]  # 이별, 긍정
# ox_2 = lyrics[(lyrics['긍부정'] == 2) & (lyrics['이별유무'] == 1)]  # 이별, 중립

DATA = lyrics.values.tolist()




def main():

    test_size = 0.3
    random_state = 13
    batch_size = 2
    EPOCHS = 10
    learning_rate = 0.0001
    num_class = 2
    max_grad_norm = 1
    warmup_ratio = 0.1
    log_interval = 200

    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)
    # device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    device = torch.device('cuda')
    print('학습을 진행하는 기기:', device)

    # title,lyrics,이별유무,긍부정,cluster_label
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
    bertmodel = AutoModel.from_pretrained("monologg/kobert")

    # tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
    #
    # bertmodel = AutoModelForMaskedLM.from_pretrained("beomi/kcbert-base")
    sents = [sent for _, sent, _, _, _ in DATA]
    labels = [label for _, _, label, _, _ in DATA]
    print(sents[:3])
    print(labels[:3])
    x_train, x_test, y_train, y_test = train_test_split(sents, labels,
                                                        test_size=test_size,
                                                        shuffle=True,
                                                        stratify=labels,
                                                        random_state=random_state)

    x_train = Build_X(x_train, tokenizer, device)
    y_train = Build_y(y_train, device)
    x_test = Build_X(x_test, tokenizer, device)
    y_test = Build_y(y_test, device)




    train_dataset = SimpleDataset(x_train, y_train)
    test_dataset = SimpleDataset(x_test, y_test)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)

    classfer = Broken_up_Classifer(bertmodel, num_class=num_class, device=device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in classfer.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in classfer.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = SGD(optimizer_grouped_parameters, lr=learning_rate)

    t_total = len(train_dataloader) * EPOCHS
    warmup_step = int(t_total * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    #------------------------------ train & test ---------------------------------
    print('학습시작')
    train_test(train_dataloader=train_dataloader,
               test_dataloader=test_dataloader,
               model=classfer,
               EPOCHS=EPOCHS,
               optimizer=optimizer,
               max_grad_norm=max_grad_norm,
               scheduler=scheduler,
               log_interval=log_interval)


if __name__ == '__main__':
    main()