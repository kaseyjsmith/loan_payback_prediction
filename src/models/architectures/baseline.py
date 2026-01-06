from src.models.architectures.base import BaseNN
from torch.nn import Dropout, Linear
import torch.nn.functional as F


class LoanPaybackNN(BaseNN):
    def __init__(
        self,
        input=30,
        h1=64,
        h2=32,
        h3=16,
        h4=8,
        out=1,
        dropout_rate=0.2,
        pos_weight=None,
        lr=0.008,
        run_id=None,
    ):
        super().__init__(pos_weight=pos_weight, lr=lr, run_id=run_id)
        self.input = Linear(input, h1)
        self.h1 = Linear(h1, h2)
        self.h2 = Linear(h2, h3)
        self.h3 = Linear(h3, h4)
        self.h4 = Linear(h4, out)

        self.dropout_rate = dropout_rate
        self.dropout = Dropout(self.dropout_rate)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = self.dropout(x)

        x = F.relu(self.h1(x))
        x = self.dropout(x)

        x = F.relu(self.h2(x))
        x = self.dropout(x)

        x = F.relu(self.h3(x))
        x = self.dropout(x)

        return self.h4(x)


if __name__ == "__main__":
    model = LoanPaybackNN()
    print(model)
