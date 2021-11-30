# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import torch
from transformers import AutoTokenizer

from Cooperation_project.Builder import Build_X


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)

    # gpu 사용 코드
    device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    # device = torch.device('cpu')
    print('학습을 진행하는 기기:', device)


    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

    DATA = pd.read_csv('text.csv')
    DATA = DATA.values.tolist()
    # sents = [sent for sent in DATA]
    sentences = []
    for sents in DATA:
        for sent in sents:
            sentences.append(sent)
    print(sentences[:3])
    x_train = Build_X(sentences, tokenizer, device)
    print(x_train[:3])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
