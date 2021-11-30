import pandas as pd
import csv
import torch
from torch.optim import Adam, SGD
from torch.utils.data.dataloader import default_collate
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

import pandas as pd

from Cooperation_project.Builder import Build_X
from DEC.pt_DEC.SimpleDataset import SimpleDataset
from DEC.pt_DEC.dec import DEC
from DEC.pt_DEC.model import train


column_name = ['lyrics','이별유무','긍부정']
# lyric = pd.read_csv("C:/Users/namva/자연어/lyrics_emotion/translate/trans_full_label.csv")
# memo = lyric.values.tolist()
# lyrics = pd.DataFrame(memo, columns=column_name)
lyrics = pd.read_csv(r'C:\Users\jeonguihyeong\PycharmProjects\Cooperation_project\DEC\pt_DEC\sibal.csv')



def main():
    # tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")
    #
    # bertmodel = AutoModelForMaskedLM.from_pretrained("monologg/kobigbird-bert-base")


    bertmodel = AutoModel.from_pretrained("monologg/kobert")
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")

    # DATA = pd.read_csv('이별유무.csv')
    # DATA = DATA.values.tolist()
    lyrics = pd.read_csv(r'C:\Users\jeonguihyeong\PycharmProjects\Cooperation_project\DEC\pt_DEC\sibal.csv')
    ox_0 = lyrics[(lyrics['긍부정'] == 0) & (lyrics['이별유무'] == 1)]  # 이별, 부정
    ox_1 = lyrics[(lyrics['긍부정'] == 1) & (lyrics['이별유무'] == 1)]  # 이별, 긍정
    ox_2 = lyrics[(lyrics['긍부정'] == 2) & (lyrics['이별유무'] == 1)]  # 이별, 중립
    # device = torch.device('cuda:0')
    device = torch.device('cpu')
    ox_1 = ox_1.values.tolist()
    sents = [sent for _,sent, _, _, _ in ox_1]
    sents_test = sents[:3]
    # sents = [sent for _, sent, _ in DATA]
    # sents = Build_X(sents, tokenizer, device=device)
    sents_test = Build_X(sents, tokenizer, device=device)
    print(sents[:3])
    dataset = SimpleDataset(sents_test)

    model = DEC(cluster_number=6, hidden_dimension=768, encoder=1)
    optimizer = SGD(model.parameters(), lr=0.001)
    index_list = train(model=model,
                       dataset=dataset,
                       epochs=3,
                       batch_size=2,
                       optimizer=optimizer,
                       stopping_delta=None,
                       collate_fn=default_collate,
                       cuda=True,
                       sampler=None,
                       silent=None,
                       update_freq=10,
                       evaluate_batch_size=16,
                       update_callback=None,
                       epoch_callback=None,
                       bertmodel=bertmodel)

    # dataset: torch.utils.data.Dataset,
    # model: torch.nn.Module,
    # epochs: int,
    # batch_size: int,
    # optimizer: torch.optim.Optimizer,
    # stopping_delta: Optional[float] = None,
    # collate_fn = default_collate,
    # cuda: bool = True,
    # sampler: Optional[torch.utils.data.sampler.Sampler] = None,
    # silent: bool = False,
    # update_freq: int = 10,
    # evaluate_batch_size: int = 1024,
    # update_callback: Optional[Callable[[float, float], None]] = None,
    # epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
    # predict(dataset=dataset,model=model,batch_size=2,collate_fn=default_collate,cuda=True,silent=None,return_actual= False,device=device, bertmodel=bertmodel)
    torch.save(model, 'model_1_test.pth')

    with open('ko_trans_1.csv', 'w', newline='', encoding="utf-8") as file:

        write = csv.writer(file)

        write.writerows([[hit] for hit in index_list])
if __name__ == '__main__':
    main()
