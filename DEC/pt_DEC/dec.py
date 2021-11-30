import torch
import torch.nn as nn
from transformers import AutoModel

from DEC.pt_DEC.cluster import ClusterAssignment
from DEC.pt_DEC.utils import To_cls


class DEC(nn.Module):
    def __init__(self,
                 cluster_number: int,
                 hidden_dimension: int,
                 alpha: float = 1.0,
                 ):

        super(DEC, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(
            cluster_number, self.hidden_dimension, alpha
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:

        bertmodel = AutoModel.from_pretrained("monologg/kobert")
        return self.assignment(To_cls(batch, bertmodel=bertmodel))

