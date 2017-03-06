from chainer import reporter
from chainer.training import extensions as E


class LossEvaluator(E.Evaluator):

    def evaluate(self):
        iterator = self.get_iterator('main')
        target = self.get_target('main')
        iterator.reset()

        loss = 0.
        for batch in iterator:
            in_arrays = self.converter(batch, self.device)
            loss = loss + target(*in_arrays)

        result = {'validation/loss': loss}
        reporter.report(result)
        return result
