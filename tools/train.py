import argparse

import chainer
from chainer import cuda
from chainer import iterators as I
from chainer import optimizers as O
from chainer import training
from chainer.training import extensions as E

from lib.data import kaggle
from lib.evaluations import accuracy
from lib.evaluations import auc
from lib.evaluations import loss
from lib.models import classifier
from lib.models import mlp


parser = argparse.ArgumentParser(
    description='Multitask Learning with PubChem dataset.')
# general
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--seed', '-s', default=0, type=int,
                    help='random seed')
# IO
parser.add_argument('--out', '-o', type=str, default='result',
                    help='Path to the output directory')
# training parameter
parser.add_argument('--batchsize', '-b', type=int, default=128)
parser.add_argument('--epoch', '-e', type=int, default=10,
                    help='The number of training epoch')
# model parameter
parser.add_argument('--unit-num', '-u', type=int, default=512,
                    help='The unit size of each layer in MLP')
args = parser.parse_args()

chainer.set_debug(True)

train, val = kaggle.get_kaggle()
train_iter = I.SerialIterator(train, args.batchsize)
val_iter = I.SerialIterator(val, args.batchsize, repeat=False, shuffle=False)

C = len(kaggle.task_names)
predictor = mlp.MLP(args.unit_num, C)
classifier = classifier.Classifier(predictor=predictor)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    classifier.to_gpu()

optimizer = O.SGD()
optimizer.setup(classifier)

updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)


trainer.extend(loss.LossEvaluator(
    val_iter, classifier, device=args.gpu))

train_iter_no_rep = I.SerialIterator(
    train, args.batchsize, repeat=False, shuffle=False)
trainer.extend(accuracy.AccuracyEvaluator(
    {'train': train_iter_no_rep, 'validation': val_iter},
    classifier, device=args.gpu))
trainer.extend(auc.AUCEvaluator(
    {'train': train_iter_no_rep, 'validation': val_iter},
    classifier, device=args.gpu))

trainer.extend(E.snapshot(), trigger=(args.epoch, 'epoch'))
trainer.extend(E.LogReport())
trainer.extend(E.PrintReport(['epoch', 'main/loss',
                              'train/accuracy', 'train/auc',
                              'validation/loss', 'validation/accuracy',
                              'validation/auc', 'elapsed_time']))
trainer.extend(E.ProgressBar())

trainer.run()
