# Introduction

*Quantitative Structure-Property Relationship* (QSAR in short) is a study of the relationship between
chemical structures and biological characteristics (e.g. toxicity) of compounds.
The goal of this example is to construct a model with a neural network
that predicts the outcome of assays.

# Dataset creation

## Data retrieval

Dataset retrieval is implemented as `get_kaggle` method in `tools/kaggle.py`.

We use the same dataset as [1] for training and validating neural networks.
Dataset from 15 assays.
We create one task from one assay, except assay No. 1851,
from which we create 5 tasks.
So, we have 19 tasks in total.

We use Substance ID (SID in short) to identify substances.

Substances in each assay have one of five labels that indicates the outcome of the assay for the substance
(Probe, Active, Inactive, Inconclusive, or Untested).
We retrieve the chemical structure of substances that has a label
Active or Inactive in at least one tasks as SDF.
Pubchem has a REST API for retrieving the dataset. We used it.

We get 56326 substances in total.

## Preprocessing

We encode labels Active and Inactive as 1 and 0, respectively and others as -1 that indicates "missing".
We convert the chemical structures of substances into fixed-length bit vector called *fingerprint*.

There are many software algorithms to encode chemical compounds to fingerprint.
In original paper [1], the authors used DRAGON to create fingerprint.
But as DRAGON is a proprietary software, we use the ECFP algorithm implemented in RDKit instead.

In summary, the preprocessed dataset is a pair of a vector of fingerprints and a vector of labels
Each fingerprint is a fixed-length bit vector.
In the context of machine leanring, the input data is called *feature*, or *feature vector*.
We treat it as a feature vector.
Each label is also a binary vector whose length is 19, which is same as the number of tasks.

## Preprocessed dataset

As data retrieval through the REST API and conversion to SMILES take long time,
we have done it and uploaded the preprocessed data.

If you want to check the retrieved dataset. You can download it from
https://www.dropbox.com/s/g25vyeralmba4d0/pubchem.h5?raw=1

# Model

## Predictor

The predictor we build in this example is a 2-layered perceptron.

```python
predictor = mlp.MLP(unit_num, C)
```

Here, `unit_num` is the number of units
`C` is the number of tasks (i.e. 19 in this example).
Note that computations are done in a *batch* manner, the output of
the predictor will have a shape `(N, C)` where `N` is the number of samples
in a minibatch.

We realize the MLP as the subclass of `chainer.ChainList`.

```python
class MLP(chainer.ChainList):

    def __init__(self, *units):
        layers = [L.Linear(None, unit) for unit in units]
        super(MLP, self).__init__(*layers)
        self.train = True

    def __call__(self, x):
        for l in self[:-1]
            x = l(x)
            x = F.relu(F.dropout(x, train=self.train))
        return self[-1](x)
```

First, it setups fully-connected(FC) layers, which are building blocks of the MLP in `__init__`.
In Chainer, the FC layer corresponds to `L.Linear`.
And it sequentially applies them in `__call__` method.
Dropout(`F.dropout`) and ReLu(`F.relu`) layers are inserted after each FC layer,
except the final one.

As the behavior of the Dropout is different in training and testing phases,
we set an attribute `train` that represents the mode of this Chain and switches
the behavior accordingly.

Q. Confirm that `predictor` defined above has two FC layers, `unit_num` units
between two FC layers, and `C` output units.

Q. Change the architecture of predictors (e.g. the number of layers or units in the layers) and check how the final accuracy changes.

## Loss function

As we want the predictor to output the probability of being active (or inactive)
given the input feature, its output should be between 0 and 1.
*sigmoid* function is suitable for this purpose.

It is defined as
`sigmoid(x) = 1 / (1 + exp(-x))`,
where `x` is a scalar (i.e. 1-dimensional float value).

Q. Draw the graph of the sigmoid function and verify that it is a monotonically
increasing function whose range is `(0, 1)`.

We train the predictor by minimizing loss values.
As the target label of each task is binary, it is common to use
the *cross entropy* loss defined as
`L(p, t) = \sum_{i=1}^{n} y_i log p_i + (1-y_i) log p_i`,
where `p` and `t` is a 1-dimensional array of length `N` whose
`i`-th value represent the predicted probability of being active
and the ground truth label for `i`-th sample.

As this is the multitask learning of `C` tasks, the actual `p` and `t` have
shape of `(N, C)` and we compute loss values for each tasks.

For numerically stable computation, it is common to combine
sigmoid function with the following cross entropy.
Chainer has `F.sigmoid_cross_entorpy` to do it.

We calculate the loss value by wrapping the predictor with `Classifier`.

```python
classifier = classifier.Classifier(predictor=predictor)
```

What it does is simply apply `F.sigmoid_cross_entropy` to calculates
and reports the loss value for this minibatch.

```python
class Classifier(chainer.Chain):

    def __call__(self, *args):
        x, t = args
        y = self.predictor(x)

        loss = F.sigmoid_cross_entropy(y, t)
        reporter.report({'loss': loss}, self)
        return loss
```

# Extension

Trainer is augmented by extending it with *Extension* in Chainer.

We can make an extension by either decorating a function with
`chainer.training.make_extension` decorator or inheriting `chainer.training.Extension`.

We use extension mechanism to:

* evaluate predictor (explained later).
* take snapshot of training process.
* save and report metrics and statistics.
* show progress bar to visualize the current progress of the training process.

We can insert the extension with `Trainer.extend` method.

```python
trainer.extend(accuracy.AccuracyEvaluator(
    {'train': train_iter_no_rep, 'validation': val_iter},
    classifier, device=args.gpu))
```


## Evaluation

There are several way to evaluate correctness of the predictor.
We use three metrics in this example, namely, loss values, accuracy, and AUC.
We compute them for both training and validation dataset to see if the model does not overfit nor underfit.

As their implementations are similar, we only look at how the accuracy is computed in detail
and left the other two for readers.
`AccuracyEvaluator`, as the name indicates, is responsible for this task.

The essential part of `AccuracyEvaluator` is in the `evaluate` method.

```python
    def evaluate(self, iterator):
        # (Omitted)
        correct, support = None, None
        for batch in iterator:
            in_arrays = self.converter(batch, self.device)
            correct_, support_ = self._evaluate_one(*in_arrays)
            correct = correct_ if correct is None else correct + correct_
            support = support_ if support is None else support + support_
        accuracy = correct.astype(np.float32) / support
        # (Omitted)
```

Here, `iterator` is an iterator that runs through either the training or the testing dataset.
We repeatedly extract a minibatch from the iterator, count the number of correct samples and the support (i.e. the number of samples that has either Activate and Inactivate labels) for each minibatch, and accumulate them to `correct` and `support` variables.
Finally, accuracy is computed with these two variables.


Q. Implement an extension `PrecisionEvaluator` that computes precision for training and test dataset.


# Software engineering

## Unit test

Once you implement the function, you need to verify if it works correctly.
*Unit test* is a common way of testing functions.

Unit tests for this examples are located in `tests` directory.

Let's take `lib.evaluations.count` method for example.

```python
def count(y, t):
    xp = cuda.get_array_module((y, t))
    mask = t != -1
    t = 2 * t - 1  # convert label 0->-1, 1->1
    correct = xp.sum(((y * t) > 0) & mask, axis=0)
    support = xp.sum(mask, axis=0)
    return correct, support
```

This method takes predicted values `y`, which is supposed to be the output
of the predictor and correct labels `t`,
both represented as 2-dimensional arrays (either NumPy or CuPy `ndarray`)
whose shape is `(N, T)` where `N` is a number of samples in a minibatch
and `T` is a number of tasks.

It counts the number of samples that correctly predict the label
and the number of samples that is not ignored for each task.

Label usually take either `0` or `1`.
`i`-th sample is ignored in `j`-th task if `t[i, j] == -1`.

Here is the test code that checks the behavior of this function.
You can see the whole example in `tests/test_accuracy.py`

```python
class TestCount(unittest.TestCase):

    def test_support(self):
        y = np.random.uniform(0, 1, (3, 4)).astype(np.float32)
        t = np.array([[0, 1, 0, 0],
                      [0, 1, 1, -1],
                      [0, 1, 0, 1]], dtype=xp.int32)
        _, support = accuracy.count(y, t)
        expect = np.array([3, 3, 3, 2], dtype=np.int32)
        np.testing.assert_array_equal(support, expect)
```

It creates sample inputs (`y` and `t`),
feeds the input to the function tested, and compares the output
with expected one.

Unit tests is useful for debugging the code.
It is sometimes beneficial to write unit tests for suspicious functions
instead of printing their outputs.

In practice, it is rarely possible to write unit tests for all functionalities
of all functions in experiments.
It is recommended to write tests for the most suspicious

You will find as you need to write unit tests that
you need to keep the function small and determine its specification.
That is one of the benefit of writing unit tests.

It is preferable that unit tests can *corner case* (a.k.a. *edge case*), or
a situation that does not occur in normal operation.
One of the corner case is that all elements in `t` being `-1`,
i.e. all samples are ignored by all tasks.

Q. What other corner cases does this function has?

Q. Write a unit test that checks that `PrecisionEvaluator` (implenented in the former question) correctly calculates precision.

## Reference

[1] Dahl, G. E., Jaitly, N., & Salakhutdinov, R. (2014). Multi-task neural networks for QSAR predictions. *arXiv preprint* arXiv:1406.1231.
