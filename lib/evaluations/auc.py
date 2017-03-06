import chainer
from chainer import functions as F
import numpy as np
import six
from sklearn import metrics

from lib.evaluations import base_evaluator as B


def compute_auc(t, p):
    mask = t != -1

    if mask.sum() == 0:
        return 0.

    t, p = t[mask], p[mask]
    return metrics.roc_auc_score(t, p)


class AUCEvaluator(B.BaseEvaluator):

    def evaluate(self, iterator):
        iterator.reset()

        t, p = [], []
        for batch in iterator:
            in_arrays = self.converter(batch, self.device)
            t_, p_ = self._evaluate_one(*in_arrays)
            t.append(t_)
            p.append(p_)
        t, p = np.vstack(t), np.vstack(p)

        auc = np.array([compute_auc(t_one, p_one)
                        for t_one, p_one in six.moves.zip(t.T, p.T)])
        result = dict(['auc_%d' % i, a]
                      for i, a in enumerate(auc))
        result['auc'] = auc.mean()
        return result

    def _evaluate_one(self, x, t):
        predictor = self.get_target('main').predictor
        y = predictor(x)
        p = F.sigmoid(y).data
        return (chainer.cuda.to_cpu(t),
                chainer.cuda.to_cpu(p))
