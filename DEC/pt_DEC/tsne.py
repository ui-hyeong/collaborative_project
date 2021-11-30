import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate

from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
# 시작
from Cooperation_project.Builder import Build_X
from DEC.pt_DEC.SimpleDataset import SimpleDataset
from DEC.pt_DEC.utils import To_cls

batch_size=2
collate_fn=default_collate
sampler=None
bertmodel = AutoModel.from_pretrained("monologg/kobert")
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
#DATA = pd.read_csv('ko_trans.csv')
#DATA = DATA.values.tolist()
device = torch.device('cpu')
DATA = pd.read_csv('sibal.csv')

fare_data = DATA[(DATA['이별유무']==1)&(DATA['긍부정']==2)]
fare_data = fare_data.values.tolist()

sents = [sent for _,sent,_,_ in fare_data]

print(sents[:2])
sents = Build_X(sents, tokenizer, device=device)
dataset = SimpleDataset(sents)
static_dataloader = DataLoader(dataset,
                               batch_size=batch_size,
                               collate_fn=collate_fn,
                               pin_memory=False,
                               sampler=sampler,
                               shuffle=False,
                               )
data_iterator = tqdm(static_dataloader,
                     leave=True,
                     unit="batch",
                     postfix={"epo": -1,
                              "acc": "%.4f" % 0.0,
                              "lss": "%.8f" % 0.0,
                              "dlb": "%.4f" % -1,},

                     )
features = []
for index, batch in enumerate(data_iterator):
    if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
        batch = batch  # if we have a prediction label, separate it to actual
    batch = batch.to(device)
    # batch = To_cls(batch, bertmodel=bertmodel).to(device)
    features.append(To_cls(batch ,bertmodel).detach().cpu())

feature = torch.cat(features).numpy()
feature2 = feature
# TSNE
tsne = TSNE(n_components=3, verbose=1, perplexity=100, random_state=42)
X_embedded = tsne.fit_transform(feature2)
print('Embedding shape 확인', X_embedded.shape)



import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style("whitegrid", {'axes.grid' : False})

fig = plt.figure(figsize=(12,12))

# ax = Axes3D(fig)
ax = fig.add_subplot(111, projection='3d') # Method 2

c = np.abs(X_embedded[:,2])
cmhot = plt.get_cmap("cool")

ax.scatter(X_embedded[:,0], X_embedded[:,1], X_embedded[:,2], c=c, cmap=cmhot, marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.title('glove, Tsne, SK-means')

ax.view_init(60,-60)

# plt.savefig("/content/drive/MyDrive/기업프로젝트/Mecab_Glove_t-SNE_SKMeans.png")
plt.show()