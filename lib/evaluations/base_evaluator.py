from chainer import reporter as reporter_
from chainer.training import extensions as E


class BaseEvaluator(E.Evaluator):

    def __call__(self, trainer=None):
        reporter = reporter_.get_current_reporter()
        result = self._evaluate()
        reporter.report(result)
        return result

    def _evaluate(self):
        result = {}
        for name, it in self._iterators.items():
            result_ = self.evaluate(it)
            result_ = dict((name + '/' + k, v) for k, v in result_.items())
            result.update(result_)
        return result
