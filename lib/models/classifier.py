import chainer
from chainer import functions as F
from chainer import reporter


class Classifier(chainer.Chain):

    def __call__(self, *args):
        x, t = args
        y = self.predictor(x)

        loss = F.sigmoid_cross_entropy(y, t)
        reporter.report({'loss': loss}, self)
        return loss
