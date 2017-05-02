# Multitask learning with multi layer perceptron for QSAR

This is an example of the application of deep learning to
Quantitative structure–activity relationship (QSAR) prediction.

The implementation is based on [1], but some minor modifications are applied.
We use [PubChem](https://pubchem.ncbi.nlm.nih.gov) as a dataset of
chemical compounds and assay outcomes, and [Chainer](http://chainer.org)
to build, train, and evaluate deep learning models.

See `commentary.md` for the detail explanation.

# Dependency

* [Chainer](http://chainer.org)
* [NumPy](http://www.numpy.org)
* [pandas](http://pandas.pydata.org)
* [RDKit](http://www.rdkit.org)
* [scikit-learn](http://scikit-learn.org/stable/)
* [six](https://pypi.python.org/pypi/six)
* [nose](http://nose.readthedocs.io/en/latest/) (for testing only)

# Usage

```
$ PYTHONPATH="." python tools/train.py
```

It runs the program on CPU by default.
If you want to run the program with GPU, add `--gpu <GPU ID>` option.
Run `python tools/train.py --help` to see the complete list of options.

Q. What is `PYTHONPATH`? Why do we need to specify it?

# Test

Run all tests including GPU ones.

```
PYTHONPATH="." nosetests tests
```

Without GPU tests

```
PYTHONPATH="." nosetests -a '!gpu' tests
```

## Reference

[1] Dahl, G. E., Jaitly, N., & Salakhutdinov, R. (2014). Multi-task neural networks for QSAR predictions. *arXiv preprint* arXiv:1406.1231.
