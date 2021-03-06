from typing import Tuple

import torch


class SimpleDataset( ):
    """
    DataLoader를 사용하기 위한.. Dataset.
    """
    def __init__(self, X: torch.Tensor):
        self.X = X


    def __len__(self) -> int:
        """
        Returning the size of the dataset
        :return:
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        """
        Returns features & the label
        :param idx:
        :return:
        """
        return self.X[idx]