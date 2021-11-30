import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")

model = AutoModelForMaskedLM.from_pretrained("monologg/kobigbird-bert-base")


device = torch.device('cpu')
print('학습을 진행하는 기기:', device)

data = '세상이 멈춘 것 같았어 우연히 널 거리에서 마주쳤을 때 가끔 들리는 너의 안부에도 난 꼭 참았는데 이 바보야 너 땜에 아프잖아 왜 또 옷은 춥게 얇게 입었어 나를 피하는 눈빛이 야윈 너의 얼굴이 그런 니가 미워서 나는 또 밤새 취해 간다 어디부터 잘못된 걸까 천천히 너에게 맞춰 기다렸다면 내가 가진 현실은 초라했고 마음만 커져가고 나 땜에 힘들다고 했잖아 행복해지고 싶다 그랬잖아 어떻게 널 보냈는데 이 바보야 너 땜에 아프잖아 왜 또 옷은 춥게 얇게 입었어 내 전부였던 눈빛이 사랑했던 얼굴이 여전히 반가워서 눈물이 흘러 어렸어서 서운해서 소중해서 불안해서 다 망쳐버린 걸 알아 다 미안해 이 바보야 얼마나 사랑했는데 어떻게 헤어졌는데 다신 만나지 말자 잡을 수 없게 잘 살아줘'

X = tokenizer(data, padding=True, truncation=True, return_tensors='pt')
y = torch.stack([
        X['input_ids'],
        X['token_type_ids'],
        X['attention_mask']
    ], dim=1).to(device)

print(y)

