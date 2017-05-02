# Introduction

*Quantitative Structure-Property Relationship* (QSAR in short) is the study of
the relationship between structures and biological characteristics
(e.g. toxicity) of chemical compounds.
Recently, Dahl et al. [1] applied deep learning to QSAR tasks and achieved
better prediction accuracy.
This work opened the door to the application of Deep Learning (DL) in bioinformatics fields.

We follow the experiment of [1], with small modifications because of some technical issues.
In this example, we will learn:

* how to prepare training/testing datasets for the task with PubChem dataset.
* how to build, train, and evaluate a DL model with Chainer.
* how to test your code with unit tests.

# Dataset creation

Dataset creation is implemented in `tools/kaggle.py`.
We describe the procedure of this function in this section.

## Retrieval

The dataset we use in this example is same as [1].
We select 15 assays from the PubChem dataset and create one task for each assay,
except assay ID (AID) 1851, from which we create 5 tasks.
Therefore, the dataset consists of 19 tasks in total.

Each substance in an assay has one of five labels (Probe, Active, Inactive, Inconclusive, or Untested)
that represents the outcome for the substance.
For each assay, we filter out compounds that have labels other than Active and Inactive and creates a task
that consists of a pair of chemical compounds and assay outcomes.
AID 1851 conducted five independent assays to the same set of substances.
We separate and treat them as five independent tasks, and applied the same filtering to them.

As one substance can occur in several tasks, we have to identify substances in different tasks.
PubChem assigns two types of IDs, namely, compound ID (CID) and substance ID (SID) to substances.
We use SID to identify them.
56326 compounds have either Active or Inactive label in at least one tasks.

The assay outcomes and chemical structure of substances are retrieved via the PubChem REST API.
Substances are converted from the [SDF file format](https://en.wikipedia.org/wiki/Chemical_table_file#SDF) to
[SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) with [RDKit](http://www.rdkit.org).
As data retrieval takes a long time, we get and store the data in [HDF5 format](https://support.hdfgroup.org/HDF5/) in advance.
If you are interested in PubChem REST API, see [the official document](https://pubchem.ncbi.nlm.nih.gov/pug_rest/PUG_REST.html) for the detailed specification.

Q. Download the dataset from the URL below and check the contents. For example, you can load HDF5 files with [`HDFStore`](http://pandas.pydata.org/pandas-docs/stable/io.html#hdf5-pytables) of Pandas.

* Dataset URL: https://www.dropbox.com/s/g25vyeralmba4d0/pubchem.h5?raw=1


## Preprocessing

Assay outcomes are encoded into *ternary* (not binary) labels.
Active and Inactive are converted as 0 and 1, respectively.
If a substance is not found in a task, or has labels other than Active
or Inactive, its task label is set to -1 which indicates "missing".

Substances are converted into fixed-length bit vectors (2048 bit in this example) called *fingerprint*.
There are many algorithms and software implementations to encode compounds to fingerprints.
In the original paper, the authors used [DRAGON](https://chm.kode-solutions.net/products_dragon.php).
However, since it is proprietary software, we use Extended Connectivity Fingerprint (ECFP) [2], which is one of the most popular encoding methods, implemented in RDKit instead.

In summary, the preprocessed dataset consists of 19 tasks, each of which
is a pair of a list of fingerprints and a list of labels.
In the context of machine learning problems, the input information (fingerprints
 in this example) which we extract from raw data and which represents the characteristics of the data is called a *feature vector*, or a *feature* in short.

# Model

## Predictor

The predictor we build in this example is a 2-layered perceptron (
is a shorthand of *multi-layer perceptron*):

```python
# tools/train.py
predictor = mlp.MLP(unit_num, C)
```

Here, `unit_num` is the number of units and `C` is the number of tasks (i.e. 19 in this example).
Note that as computations are done in a *minibatch* manner, the output of
the predictor has a shape `(N, C)` where `N` is the number of samples
in a minibatch.

We realize the MLP as the subclass of [`chainer.ChainList`](http://docs.chainer.org/en/stable/reference/core/link.html#chainer.ChainList):

```python
# lib/models/mlp.py
class MLP(chainer.ChainList):

    def __init__(self, *out_units):
        layers = [L.Linear(None, unit) for unit in out_units]
        super(MLP, self).__init__(*layers)
```

In `__init__` method, it sets up fully-connected (FC) layers, which are building blocks of the MLP.
Chainer prepares the FC layer as [`L.Linear`](http://docs.chainer.org/en/stable/reference/links.html#chainer.links.Linear).
In this example (and also in the document of Chainer), we use `L` as the abbreviation of `chainer.links`.

The forward propagation of `MLP` is defined in its `__call__` method:

```python
# lib/models/mlp.py
class MLP(chainer.ChainList):

    # (Omitted)

    def __call__(self, x):
        for l in self[:-1]
            x = F.relu(l(x))
        return self[-1](x)
```

It sequentially applies layers defined above.
ReLu ([`F.relu`](http://docs.chainer.org/en/stable/reference/functions.html#chainer.functions.relu)) layers are inserted after each FC layer, except the final one (Similarly to `L`, `F` is an alias of `chainer.functions`).
Note that in the original paper, the authors apply Dropout after FC layers. But we do not use it and left it to the readers. 

Q. Confirm that `predictor` defined above has two FC layers, `unit_num` units
between two FC layers and `C` output units.

Q. In the original paper, the authors used an MLP consisting of up to three FC layers. Change the predictor from two-layer to three-layer and check how the final accuracy changes. Try architectures other than MLP.

Q. Why should we not add ReLu to the final FC layer? See what happens if we do that.

Q. In this question, we add Dropout between FC layers and ReLU layers. Be aware that the behavior of Dropout is different during training and test phases. So, we have to change the behavior of the model accordingly. 
We introduce Drop by following these steps:

1. Insert Dropout in the definition of the forward propagation implemented in `Model.__call__`. Dropout is implemented as [`F.dropout`](http://docs.chainer.org/en/stable/reference/functions.html#chainer.functions.dropout) in Chainer.
2. Add `train` attribute to `Model`, which specifies the mode of the model. `train` should be set to `True` initially.
3. Each `Evaluator` (explained later) extracts the model to evaluate it during test phase (e.g. see [here](https://github.com/delta2323/BMI219-2017-DeepQSAR/blob/master/lib/evaluations/accuracy.py#L43), `predictor` is a instance of `Model`). Set the `train` attribute of the model to `False` temporaliry before forward propagation.
4. After the forward propagation. Set the `train` attribute to `True` again.

Hint: [`TestModeEvaluator`](https://github.com/pfnet/chainer/blob/master/examples/imagenet/train_imagenet.py#L68) in the official ImageNet example could be helpful.


## Sigmoid function

We want the predictor to output the probability of being active
given the input feature. Therefore, its output should be between 0 and 1.
*Sigmoid function* is suitable for this purpose. It is defined as

```
sigmoid(x) = 1 / (1 + exp(-x)),
```

where `x` is a scalar float value.
When we apply the sigmoid function to a tensor, or multi-dimensional array,
it is common to apply the scalar function in element-wise manner.

Q. Draw a graph of the sigmoid function and verify that it is a monotonically
increasing function whose range is `(0, 1)`.

## Cross entropy loss

We train the predictor by minimizing some predefined loss values.
As target labels are binary, it is common to use
a *cross entropy loss* defined as

```
cross_entropy(p, t) = \sum_{i=1}^N y_i \log p_i + (1 - y_i) \log (1 - p_i),
```

where `p` and `t` is a 1-dimensional array of length `N` (or a minibatch) whose
`i`-th value represent the predicted probability of being active
and the ground truth label for `i`-th sample, respectively.
As it is a multitask learning of `C` tasks, the actual `p` and `t` have shapes of `(N, C)`.
We compute the cross entropy loss for each task and sum them up to get the final loss.

The loss value is calculated by wrapping the predictor with `Classifier`.

```python
# tools/train.py
classifier = classifier.Classifier(predictor=predictor)
```

`Classifier` is a subclass of `Chain`.
Its `__call__` method takes the both the predictor input and the target output for a minibatch, calculates the loss value, and reports it:

```python
# lib/models/classifier.py
class Classifier(chainer.Chain):

    def __call__(self, *args):
        x, t = args
        y = self.predictor(x)
        loss = F.sigmoid_cross_entropy(y, t)
        reporter.report({'loss': loss}, self)
        return loss
```

You might notice that we use [`F.sigmoid_cross_entorpy`](http://docs.chainer.org/en/stable/reference/functions.html#chainer.functions.sigmoid_cross_entropy)
in `__call__` instead of applying the singmoid and cross entropy functions separately.
It is commmon in most deep learning frameworks to combine them when possible to improve numerical stability.


# Evaluation

## Extension

Trainers are augmented with *Extensions* in Chainer.
We can make an extension by either decorating a function with the
`chainer.training.make_extension` decorator
or inheriting `chainer.training.Extension` class.

This example uses the extension mechanism to:

* evaluate predictor (explained later).
* take snapshot of training process.
* save and report metrics and statistics.
* show progress bar to visualize the current progress of the training process.


## Accuracy evaluation

There are several ways to evaluate the correctness of predictors.
We use three metrics in this example, namely, *accuracy*, *loss value*, and *AUC*.
As their implementations are similar, we only look at how the accuracy is computed in detail
and left the other two for readers.
`AccuracyEvaluator` is, as the name indicates, responsible for this task.

The essential part of `AccuracyEvaluator` is in the `evaluate` method:

```python
# lib/evaluations/accuracy.py

class AccuracyEvaluator(B.Evaluator):

    # (Omitted)

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
The extension repeatedly extracts a minibatch from the iterator, counts the number of correct samples and the support (i.e. the number of samples that has either activate and inactivate labels) for the minibatch in `_evaluate_one` method, and accumulates them to intermediate variables.
After consuming all minibatches, the accuracy is computed with these two variables.
Note that this extension calculates accuracies for each tasks.
So, `correct`, `support`, and `accuracy` are 1-dimensional ndarray that has as many elememnts as tasks.

We insert extensions with `Trainer.extend` method.
The metrics are computed for both training and validation dataset
to watch the model does not overfit nor underfit:

```python
# tools/train.py
trainer.extend(accuracy.AccuracyEvaluator(
    {'train': train_iter_no_rep, 'validation': val_iter},
    classifier, device=args.gpu))
```

Q. Check `AcuracyEvaluator` and `LossEvaluator` to understand how the corresponding metrics are evaluated.

Q. Implement an extension `PrecisionEvaluator` that computes precision for the training and test datasets.


# Unit testing

Once you implement a function, you need to verify that it works correctly.
*Unit tests* are a common way of testing functions.
Briefly departing from the previous sections, we explain a software engineering aspect of machine learning in this section.

We use [nose](http://nose.readthedocs.io/) for unit testing.
nose can be installed via `pip` command:

```bash
$ pip install nose
```

We only explain a simple example of unit testing here.
See the official `nose` documentation for the details.

## Example: count function

Let's take `lib.evaluations.count` function for example:

```python
# lib/evaluations/accuracy.py
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
Both are represented as 2-dimensional arrays of either NumPy or CuPy `ndarray`
whose shape is `(N, C)` where `N` is a number of samples in a minibatch
and `C` is a number of tasks.

It counts the number of samples whose label is correctly predicted (`correct`)
and the number of non-ignored samples (`support`) for each task.
The predicted value is the sign of `y`.
`t` usually take either `0` or `1`, which represent active and inactive,
respectively, but `i`-th sample is *ignored* in `j`-th task if `t[i, j] == -1`.


## Unit testing code for the count function

Here is unit testing code that checks the behavior of `count` function.
All tests are located in `tests` directory and this unit test is in `tests/test_accuracy.py`.

```python
# tests/test_accuracy.py
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
feeds the input to the function to be tested, and compares the output
with the expected one.
`np.testing.assert_array_equals` compares thow NumPy ndarrays
and raises an error if they do not agree within the specified precision.


## Run the test code

To run the test with `nose`, type the following command in a terminal:

```
$ nosetests tests/test_accuracy.py
```

It runs all tests in the file (see [the document](http://nose.readthedocs.io/en/latest/writing_tests.html#writing-tests)
if you are interested in )
You will get a result something like this if all tests pass:

```
$ nosetests tests/test_accuracy.py                                            
......
----------------------------------------------------------------------
Ran 6 tests in 6.010s

OK
```

If your environment does not support GPU, add `-a '!gpu'` to the command.
See [the corresponding part](http://nose.readthedocs.io/en/latest/plugins/attrib.html#module-nose.plugins.attrib) of the nose document for detail.
 

Q. `count` assumes that `y` is a 2-dimensional float array. But we cannot guarantee that users
always feed valid inputs. We need to check if the inputs are the expected ones. Change the `count` method so that it raises [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) when `y` does not meet the condition above and write a test case that checks this input validation works correctly (Hint: you can use [`unittest.assertRaises`](https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertRaises) to check the function raises an expected error).


## Corner cases

It is preferable that unit tests check *corner cases* (a.k.a. *edge cases*), or
a situation that does not occur in normal operation.

Q.  One of the corner cases the `count` method has is that all values in a column of `t` are `-1`,
that is, all samples are ignored by some task.
In such a case, we cannot calculate the accuracy because the support is 0, causing division-by-zero.
We determine that if all samples are ignored by some task, the function should return `numpy.inf` as the accuracy for the task 
Write a test case that checks this behavior. You might need to modify `count` method.

Q. What other corner cases does this function have?

Q. Write a unit test that checks that `PrecisionEvaluator` (implenented in the former question) correctly calculates precision.


## Benefits of unit tests 

One of the benefits of writing unit tests is its usefulness for debugging.
You may often run code and debug-print intermediate outputs to debug them.
Although it is easy and powerful, it is getting harder to find bugs solely with this
technique as the code size grows.
You can verify the behavior of functions (although not perfectly)
and narrow down where bugs reside by writing unit tests for suspicious functions.

Unit tests are also beneficial to keep your code clean and tidy.
You will find as you write unit tests that
it is difficult to write unit tests for huge and monolithic functions.
Also you may notice that you cannot write test code unless you clearly determine
the behavior of the target functions.
If we want to write unit tests appropriately, functions tend to be small and have clear-cut specification.
That is another positive side effect of unit testing.


## Practical considerations

In practice, however, it is rarely possible to write unit tests for all functionalities
of all functions in experimental codes.
This is because contrary to library codes, the code for experiments is subject to change because
of many trial & errors.
So, it is recommended to write tests for the most suspicious and unconfident part if your coding time is limited.


## Reference

[1] Dahl, G. E., Jaitly, N., & Salakhutdinov, R. (2014). Multi-task neural networks for QSAR predictions. *arXiv preprint* arXiv:1406.1231.

[2] Rogers, David, and Mathew Hahn. "Extended-connectivity fingerprints." Journal of chemical information and modeling 50.5 (2010): 742-754.
