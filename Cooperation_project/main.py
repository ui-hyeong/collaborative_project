from typing import List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import optim
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader


from transformers.optimization import get_cosine_schedule_with_warmup, AdamW

# device = torch.device("cuda:0")
from Cooperation_project.Builder import Build_X, Build_y
from Cooperation_project.SimpleDataset import SimpleDataset
from Cooperation_project.EmoClassifer import EmoClassifer
from Cooperation_project.train_test import train_test

DATA = pd.read_csv('C:\Users\jeonguihyeong\PycharmProjects\Cooperation_project\bi_classifer\DEC_cluster.csv')
DATA = DATA[(DATA['이별유무'] == 1)]
DATA = DATA.values.tolist()

# print(DATA[:3])
# print(TARGET[:3])


def main():

    test_size = 0.3
    random_state = 42
    batch_size = 64
    EPOCHS = 3
    learning_rate = 5e-5
    num_class = 28
    max_grad_norm = 1
    warmup_ratio = 0.1
    log_interval = 200

    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)

    # device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    device = torch.device('cpu')
    print('학습을 진행하는 기기:', device)
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")

    model = AutoModelForMaskedLM.from_pretrained("monologg/kobigbird-bert-base")
    # sents = [sent for sent in DATA]
    sentences = []
    for sents in DATA:
        for sent in sents :
            sentences.append(sent)

    print(sentences[:3])
    # labels = [label for _, label in DATA]
    x_train, x_test, y_train, y_test = train_test_split(sentences, TARGET,
                                                        test_size=test_size,
                                                        shuffle=True,
                                                        random_state=random_state)
    print(x_train[:3])
    x_train = Build_X(x_train, tokenizer, device)
    y_train = Build_y(y_train, device)
    x_test = Build_X(x_test, tokenizer, device)
    y_test = Build_y(y_test, device)
    print(x_train[:3])



    train_dataset = SimpleDataset(x_train, y_train)
    test_dataset = SimpleDataset(x_test, y_test)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)

    classfer = EmoClassifer(bertmodel, num_class=num_class, device=device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in classfer.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in classfer.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

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
