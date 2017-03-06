import chainer
from chainer import functions as F
from chainer import links as L


class MLP(chainer.ChainList):

    def __init__(self, *units):
        layers = [L.Linear(None, unit) for unit in units]
        super(MLP, self).__init__(*layers)
        self.train = True

    def __call__(self, x):
        for l in self[:-1]:
            x = l(x)
            x = F.relu(F.dropout(x, train=self.train))
        return self[-1](x)
