import chainer
from chainer import cuda
import numpy as np

from lib.evaluations import base_evaluator as B


def count(y, t):
    xp = cuda.get_array_module((y, t))
    mask = t != -1
    t = 2 * t - 1  # convert label 0->-1, 1->1
    correct = xp.sum(((y * t) > 0) & mask, axis=0)
    support = xp.sum(mask, axis=0)
    return correct, support


class MultitaskBinaryAccuracy(chainer.Function):

    def forward(self, inputs):
        correct, support = count(*inputs)
        return correct.astype(np.float32) / support,


class AccuracyEvaluator(B.BaseEvaluator):

    def evaluate(self, iterator):
        iterator.reset()

        correct, support = None, None
        for batch in iterator:
            in_arrays = self.converter(batch, self.device)
            correct_, support_ = self._evaluate_one(*in_arrays)
            correct = correct_ if correct is None else correct + correct_
            support = support_ if support is None else support + support_
        accuracy = correct.astype(np.float32) / support

        result = dict([('accuracy_%d' % i, acc)
                       for i, acc in enumerate(accuracy)])
        result['accuracy'] = accuracy.mean()
        return result

    def _evaluate_one(self, x, t):
        predictor = self.get_target('main').predictor
        y = predictor(x)
        correct, support = count(y.data, t)
        return (chainer.cuda.to_cpu(correct),
                chainer.cuda.to_cpu(support))
