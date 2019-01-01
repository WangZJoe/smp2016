import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, n_features, n_hidden, n_out):
        super(Mlp, self).__init__()
        self.hidden = nn.Linear(n_features, n_hidden, bias=True)
        self.out = nn.Linear(n_hidden, n_out, bias=True)

    def forward(self, x)：
        x = F.sigmoid(self.hidden(x))
        x = self.out(F.sigmoid(x))
        
        return x


class smp():
    def __init__(self, n_features, n_hidden, n_out):
        self.mlp = Mlp(n_features, n_hidden, n_out)
        self.optimizer = torch.optim.SGD(self.mlp.parameters(), lr=0.03)

        
    def train(self, x, label):
        fe = self.mlp(x)_
        gender = F.log_softmax(fe, dim=2)
        locate = F.log_softmax(fe, dim=8)
        age = F.log_softmax(fe, 3)

        pred_label = torch.cat((gender,locate, age), dim=1)

        loss = nn.MSELoss(pred_label, label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    