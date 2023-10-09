import torch
import torch.nn as nn
import torch.nn.functional as F



class SeNetBlock(nn.Module):
    def __init__(self, num_feats, ratio=4):
        super().__init__()

        self.num_feats = num_feats
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.senet = nn.Sequential(
            nn.Linear(num_feats, num_feats // ratio),
            nn.ReLU(),
            nn.Linear(num_feats // ratio, num_feats, bias=False),
        )

    def forward(self, x):
        '''
        param x: input data x, shape (BATCH_SIZE, NUM_FEATS, EMB_DIM), dtype torch.float32
        
        return: output data, shape (BATCH_SIZE, NUM_FEATS, EMB_DIM), dtype torch.float32
        '''
        w = self.senet(self.pool(x).squeeze(-1))
        w = torch.sigmoid(w) * 2  # 保证scale的稳定
        return w.reshape(-1, self.num_feats, 1) * x


class CancelOutBlock(nn.Module):
    def __init__(self, num_feats, dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.w = nn.parameter.Parameter(torch.randn((num_feats, 1)) * 2)

    def regularization_loss(self, reg_type='l1'):
        if reg_type == 'l1':
            return torch.sum(torch.abs(self.w.data)), torch.var(self.w.data)
        elif reg_type == 'l2':
            return torch.sum(torch.pow(self.w.data, 2)), torch.var(self.w.data)
        else:
            raise ValueError(f"Regularizer `{reg_type}` not Exists")

    def forward(self, x):
        '''
        param x: input data x, shape (BATCH_SIZE, NUM_FEATS, EMB_DIM), dtype torch.float32
        
        return: output data, shape (BATCH_SIZE, NUM_FEATS, EMB_DIM), dtype torch.float32
        '''
        w = F.dropout(self.w, p=self.dropout_rate, training=self.training)
        return 2 * torch.sigmoid(w) * x


if __name__ == '__main__':
    def test_co():
        mock_ninput = torch.tensor(torch.ones((64, 10, 32)), dtype=torch.float32)
        co = CancelOutBlock(10)
        print(co(mock_ninput))
        print(co.regularization_loss())


    def test_senet():
        mock_ninput = torch.tensor(torch.ones((64, 10, 32)), dtype=torch.float32)
        se = SeNetBlock(10)
        print(se(mock_ninput))

    test_co()
    test_senet()
