import unittest

import torch
from chowder_weak_supervised.loss.chowder_loss import LossScore
import torch.nn as nn


class TestLoss(unittest.TestCase):
    def test_loss_score(self):
        loss = LossScore()

        batch_0 = torch.FloatTensor([[0, 1, 0.3]])
        batch_1 = torch.FloatTensor([[1, 0.1, 1]])
        batch_2 = torch.FloatTensor([[0.7, 0.5, 5]])
        batch_3 = torch.FloatTensor([[0.35, 0.25, 1.4]])
        self.assertGreater(loss(batch_1, batch_3), loss(batch_0, batch_2))

    def test_BCE_score(self):
        loss = nn.BCELoss()

        batch_0 = torch.FloatTensor([[0], [0]])
        batch_1 = torch.FloatTensor([[1], [1]])
        batch_2 = torch.FloatTensor([[0.5], [0.5]])
        batch_3 = torch.FloatTensor([[0.25], [0.25]])

        # Loss for a same batch
        self.assertEqual(loss(batch_0, batch_0), 0.0)
        self.assertEqual(loss(batch_1, batch_1), 0.0)
        self.assertEqual(
            loss(batch_0, batch_1), 100.0
        )  # Pytorch assumes that log(0)=-100
        self.assertEqual(
            loss(batch_1, batch_0), 100.0
        )  # Pytorch assumes that log(0)=-100
        self.assertGreater(loss(batch_0, batch_2), loss(batch_0, batch_3))
