"""Chowder model init module."""

import torch


class CHOWDER(torch.nn.Module):
    """Implmentation of Chowder model as introduced in the paper (https://arxiv.org/pdf/1802.02212.pdf).

    # The model is implemented except for the min-max layer, which is up to the candidate.
    #     self.model = CHOWDER(self.n_extreme)
    Args:
        dropout_rate (float, optional): Drop out rate of the dropout layers. Defaults to 0.5.
        mlp_layer_1_size (int, optional): Size of the first linear layer. Defaults to 200.
        mlp_layer_2_size (int, optional): Size of the second linear layer. Defaults to 100.
        n_extreme (int, optional): Number of extreme tiles in the min-max layer. Defaults to 5.

    """

    def __init__(
        self,
        dropout_rate: float = 0.5,
        mlp_layer_1_size: int = 200,
        mlp_layer_2_size: int = 100,
        n_extreme: int = 5,
    ):
        super().__init__()
        self._n_extreme = n_extreme
        self.conv1d = torch.nn.Linear(in_features=2048, out_features=1, bias=False)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self._n_extreme * 2, out_features=mlp_layer_1_size
            ),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(
                in_features=mlp_layer_1_size, out_features=mlp_layer_2_size
            ),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(in_features=mlp_layer_2_size, out_features=1),
            # no final sigmoid as we are using torch.nn.BCEWithLogitsLoss for the criterion
        )

    def minmax(self, x):
        """Implementation of minmax layer to extract top instances region but also the negative evidences

        Check section 2.3 in CHOWDER article.
        # min-max layer: keep only the `self._n_extreme` top tiles (i.e. with the
        # highest score) and
        # the `self._n_extreme` bottom tiles (i.e. with the lowest score) for each
        # sample.

        Args
            x : output of the one-dimensional convolution layer

        Returns
            Tensor of sizer (2 * self._n_extreme, 1)

        """
        sorted_x, _ = torch.sort(x)

        return torch.cat(
            (sorted_x[: self._n_extreme], sorted_x[-self._n_extreme :]), dim=0
        )

    def forward(self, x):
        """Forward method of the Chowder model.

        Args:
            x (torch.Tensor): tensor of shape (n_samples, n_tiles, n_features)

        Returns:
            torch.Tensor: prediction tensor of shape (n_samples)

        """
        x = self.conv1d(x)
        x = x.squeeze(dim=-1)
        x = self.minmax(x)
        logits = self.mlp(x)
        return logits
