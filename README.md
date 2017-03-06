# Multitask learning for QSAR prediction

This is an example of multitask learning for QSAR prediction.
This example is based on [1], but some minor modification is applied.

We use Chainer to build, train, and evaluate neural networks.
We use PubChem as a dataset of assay outcomes and chemical structure of compounds.


See doc.md for detail explanation of this example.


# Dependency

* Chainer
* RDKit
* NumPy
* Pandas
* sci-kit learn
* Nose (for testing)


# Usage

We can run the code with the following command.

```
$ PYTHONPATH="." python tools/train.py
```

If we want to run the code with GPU, add `--gpu <GPU ID>` to the command.
Run `python tools/trian.py --help` to see the complete list of options.

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
