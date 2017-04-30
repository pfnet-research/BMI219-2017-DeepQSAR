# Introduction

*Quantitative Structure-Property Relationship* (QSAR in short) is a study of
relationship between structures and biological characteristics
(e.g. toxicity) of chemical compounds.
Recently, Dahl et al. [1] applied deep learning to QSAR tasks and achieved
better prediction accuracy.
This work opened the door of application of Deep Learning (DL) to bioinformatics fields.

We follow the experiment of [1], with small modifications because of some technical issues.
In this example, we will learn

* how to prepare training/testing datasets for the task with PubChem dataset.
* how to build, train, and evaluate DL model with Chainer.
* how to test your code with unit tests.

# Dataset creation

Dataset creation is implemented in `tools/kaggle.py`.
We will describe the procedure of this function in this section.

## Retrieval

The dataset we use in this example is same as [1].
We select 15 assays in the PubChem dataset and create one task per one assay,
except assay ID (AID) 1851, from which we create 5 tasks.
So, the dataset consists of 19 tasks in total.

Each task consists of a pair of chemical compounds and assay outcomes.
In one assay, each compound has one of five labels that represents the outcome for the substance (Probe, Active, Inactive, Inconclusive, or Untested).
For each task, we filter out compounds that has labels other than Active and Inactive.
Assay no. 1851 contains five independent assays to the same set of compounds.
We separate and treat them as five independent tasks.

As one compound can occur in several tasks.
We have to identify substances in different tasks.
PubChem assigns two types of IDs, namely, compound ID (CID) and substance ID (SID) to compounds.
We use SID to identify them and 56326 compounds have either Active or Inactive
label in at least one tasks.
We retrieve the chemical structure of compounds in [SDF file format](https://en.wikipedia.org/wiki/Chemical_table_file#SDF) and convert
it to [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system).
with [RDKit](http://www.rdkit.org).

Pubchem has a REST API for retrieving its dataset.
But as data retrieval through the REST API takes long time,
we retrieved the dataset in advance and store the data in [HDF5 format](https://support.hdfgroup.org/HDF5/).
If you want to check the raw dataset. You can download it from
[here](https://www.dropbox.com/s/g25vyeralmba4d0/pubchem.h5?raw=1).
If you are interested in PubChem REST API, see [the official document](https://pubchem.ncbi.nlm.nih.gov/pug_rest/PUG_REST.html)
 for the detail specification.

## Preprocessing

We encode labels into ternary values.
Active and Inactive are converted as 0 and 1, respectively.
If a compound is not found in some task, or has labels other than Active
or Inactive, its label corresponds to the task is set to -1 that indicates "missing".

Compounds are converted into fixed-length bit vectors called *fingerprint*.
There are many algorithms and softwares to encode compounds to fingerprint.
In the original paper [1], the authors used [DRAGON](https://chm.kode-solutions.net/products_dragon.php).
But as it is a proprietary software, we use the ECFP algorithm implemented in RDKit instead.

In summary, the preprocessed dataset consists of 19 tasks, each of which
is a pair of a list of fingerprints and a list of labels.
In the context of machine learning, the input data fed to the model (fingerprint
  in this context) is called *feature*, or *feature vector*.

# Model

## Predictor

The predictor we build in this example is a 2-layered perceptron.

```python
predictor = mlp.MLP(unit_num, C)
```

Here, `unit_num` is the number of units and
`C` is the number of tasks (i.e. 19 in this example).
Note that computations are done in a *minibatch* manner, the output of
the predictor will have a shape `(N, C)` where `N` is the number of samples
in a minibatch.

We realize the MLP as the subclass of `chainer.ChainList`:

```python
class MLP(chainer.ChainList):

    def __init__(self, *units):
        layers = [L.Linear(None, unit) for unit in units]
        super(MLP, self).__init__(*layers)
        self.train = True
```

First, it setups fully-connected(FC) layers, which are building blocks of the MLP in `__init__`.
In Chainer, the FC layer corresponds to `L.Linear`.

The forward propagation of `MLP` is defined in its `__call__` method:

```
    def __call__(self, x):
        for l in self[:-1]
            x = l(x)
            x = F.relu(F.dropout(x, train=self.train))
        return self[-1](x)

```

It sequentially applies them in `__call__` method.
Dropout(`F.dropout`) and ReLu(`F.relu`) layers are inserted after each FC layer, except the final one.

As the behavior of the Dropout is different in training and testing phases,
we set an attribute `train` that represents the mode of this Chain and switches
the behavior accordingly.

Q. Confirm that `predictor` defined above has two FC layers, `unit_num` units
between two FC layers, and `C` output units.

Q. Change the architecture of predictors (e.g. the number of layers or units in the layers) and check how the final accuracy changes.

## Sigmoid function

As we want the predictor to output the probability of being active (or inactive)
given the input feature, its output should be between 0 and 1.
*Sigmoid function* is suitable for this purpose. It is defined as

```
sigmoid(x) = 1 / (1 + exp(-x)),
```

where `x` is a scalar float value.

Q. Draw the graph of the sigmoid function and verify that it is a monotonically
increasing function whose range is `(0, 1)`.

When we apply the sigmoid function to a tensor, or multi-dimensional array,
it is common to apply the scalar function in element-wise manner.

## Cross entropy loss

We train the predictor by minimizing some predefined loss values.
As target labels are binary, it is common to use
the *cross entropy* loss defined as

```
cross_entropy(p, t) = \sum_{i=1}^N y_i \log p_i + (1 - y_i) \log (1 - p_i),
```

where `p` and `t` is a 1-dimensional array of length `N` whose
`i`-th value represent the predicted probability of being active
and the ground truth label for `i`-th sample.
As this is multitask learning of `C` tasks, the actual `p` and `t` have shape of `(N, C)`.
We compute the cross entropy loss for each task and sum them up to get the final loss.

The loss value is calculated by wrapping the predictor with `Classifier`.

```python
classifier = classifier.Classifier(predictor=predictor)
```

What it does is simply to calculate loss value and to report it for this minibatch.

```python
class Classifier(chainer.Chain):

    def __call__(self, *args):
        x, t = args
        y = self.predictor(x)

        loss = F.sigmoid_cross_entropy(y, t)
        reporter.report({'loss': loss}, self)
        return loss
```

For numerically stable computation, it is common to combine
the sigmoid function with the following cross entropy.
Chainer implement this functionality in `F.sigmoid_cross_entorpy`.


# Evaluation

## Extension

Trainers are augmented by extending them with *Extension* in Chainer.
We can make an extension by either decorating a function with
`chainer.training.make_extension` decorator
or inheriting `chainer.training.Extension` class.

This example uses this extension mechanism to:

* evaluate predictor (explained later).
* take snapshot of training process.
* save and report metrics and statistics.
* show progress bar to visualize the current progress of the training process.


## Accuracy evaluation

There are several way to evaluate correctness of models.
We use three metrics in this example, namely, accuracy, loss values, and AUC.

As their implementations are similar, we only look at how the accuracy is computed in detail
and left the other two for readers.
`AccuracyEvaluator` is, as the name indicates, responsible for this task.

The essential part of `AccuracyEvaluator` is in the `evaluate` method.

```python
    def evaluate(self, iterator):
        # (Omitted)
        correct, support = 0., 0.
        for batch in iterator:
            in_arrays = self.converter(batch, self.device)
            correct_, support_ = self._evaluate_one(*in_arrays)
            correct = correct + correct_
            support = support + support_
        accuracy = correct.astype(np.float32) / support
        # (Omitted)
```

Here, `iterator` runs through either the training or the testing dataset.
The extension repeatedly extracts a minibatch from the iterator, counts the number of correct samples and the support (i.e. the number of samples that has either activate and inactivate labels) for the minibatch, and accumulates them to intermediate variables.
After consuming all minibatches, the accuracy is computed with these two variables.

We insert extensions with `Trainer.extend` method.
The metrics are computed for both training and validation dataset
to watch the model does not overfit nor underfit.

```python
trainer.extend(accuracy.AccuracyEvaluator(
    {'train': train_iter_no_rep, 'validation': val_iter},
    classifier, device=args.gpu))
```

Q. Implement an extension `PrecisionEvaluator` that computes precision for training and test dataset.


# Unit test

In this section, we briefly explain a software engineering aspect of machine learning.
Once you implement a function, you need to verify that it works correctly.
*Unit test* is a common way of testing functions.

## Testing count function

Let's take `lib.evaluations.count` for example:

```python
def count(y, t):
    xp = cuda.get_array_module((y, t))
    mask = t != -1
    t = 2 * t - 1  # convert label 0->-1, 1->1
    correct = xp.sum(((y * t) > 0) & mask, axis=0)
    support = xp.sum(mask, axis=0)
    return correct, support
```

This function takes `y`, which is supposed to be the output
of the predictor (before applying the sigmoid function) and target labels `t`.
Both are represented as 2-dimensional arrays (either NumPy or CuPy `ndarray`)
whose shape is `(N, C)` where `N` is a number of samples in a minibatch
and `C` is a number of tasks.

It counts the number of samples whose label is correctly predicted (`correct`)
and that are not ignored (`support`) for each task.
The predicted value is the sign of `y`.
`t` usually take either `0` or `1`, which represent active and inactive,
respectively, but `i`-th sample is *ignored* in `j`-th task if `t[i, j] == -1`.

Here is a test code that checks the behavior of this function.
All tests are located in `tests` directory.

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

It is preferable that unit tests can support *corner case* (a.k.a. *edge case*), or
a situation that does not occur in normal operation.

Q. One of corner cases is that all elements in `t` being `-1`,
i.e. all samples are ignored by all tasks. In such a case, we cannot calculate accuracy
because support is 0. We define the specification of this function as returning `np.inf`
when all samples are ignored. Write a test case that checks this behavior. You might need
to modify `count` method.

Q. `count` assumes that `y` is a 2-dimensional float array. But we cannot guarantee that users
always feed valid inputs. Write a check check We need to check if the input is expected one. Change `count` method so that it raises `ValueError` when `y` does not meet the condition above and write a test case that checks this input validation works correctly.

Q. What other corner cases does this function has?

Q. Write a unit test that checks that `PrecisionEvaluator` (implenented in the former question) correctly calculates precision.


## Benefit and practical consideration of unit tests.

Unit tests are useful for debugging codes.
One of the most easiest way to debug codes is to run them and debug-print
intermediate outputs.
But as the codes grow, it is getting harder to find bugs solely with this
technique.
You can verify the behavior of functions (although not perfectly)
and narrow down where bugs reside by writing unit tests for suspicious functions.

You will find as you write unit tests that
it is difficult to write unit tests for huge and monolithic functions.
You will also notice that you cannot write test codes unless you clearly determine
the specification of the target functions.
Unit tests are sometimes beneficial to keep your codes clean and clear.
That is another positive side effect of unit testing.

In practice, however, it is rarely possible to write unit tests for all functionalities
of all functions in experimental codes.
So, it is recommended to write tests from the most suspicious and unconfident part.

## Reference

[1] Dahl, G. E., Jaitly, N., & Salakhutdinov, R. (2014). Multi-task neural networks for QSAR predictions. *arXiv preprint* arXiv:1406.1231.