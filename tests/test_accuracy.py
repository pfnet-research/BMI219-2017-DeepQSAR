import unittest

from chainer import cuda
from nose.plugins import attrib
import numpy as np

from lib.evaluations import accuracy


class TestCount(unittest.TestCase):

    def check_support(self, xp):
        y = xp.random.uniform(0, 1, (3, 4)).astype(xp.float32)
        t = xp.array([[0, 1, 0, 0],
                      [0, 1, 1, -1],
                      [0, 1, 0, 1]], dtype=xp.int32)
        _, support = accuracy.count(y, t)
        np.testing.assert_array_equal(cuda.to_cpu(support),
                                      np.array([3, 3, 3, 2], dtype=np.int32))

    def test_support_cpu(self):
        self.check_support(np)

    @attrib.attr('gpu')
    def test_support_gpu(self):
        self.check_support(cuda.cupy)

    def check_correct(self, xp):
        y = xp.array([[-.5, -.5, .5],
                      [.5, .5, .5]], dtype=xp.float32)
        t = xp.array([[0, 1, -1],
                      [1, 0, -1]], dtype=xp.int32)
        correct, _ = accuracy.count(y, t)
        np.testing.assert_array_equal(cuda.to_cpu(correct),
                                      np.array([2, 0, 0], dtype=np.float32))

    def test_correct_cpu(self):
        self.check_correct(np)

    @attrib.attr('gpu')
    def test_correct_gpu(self):
        self.check_correct(cuda.cupy)


class TestMultiTaskBinaryAccuracy(unittest.TestCase):

    def setUp(self):
        self.f = accuracy.MultitaskBinaryAccuracy()
        self.y = np.array([[-.5, -.5, .5],
                           [.5, .5, .5]], dtype=np.float32)
        self.t = np.array([[0, 1, 0],
                           [1, 0, 1]], dtype=np.int32)

    def check_forward(self, xp):
        y = xp.asarray(self.y)
        t = xp.asarray(self.t)
        acc = self.f(y, t)
        np.testing.assert_array_equal(cuda.to_cpu(acc.data),
                                      np.array([1., 0., .5], dtype=np.float32))

    def test_forward_cpu(self):
        self.check_forward(np)

    @attrib.attr('gpu')
    def test_forward_gpu(self):
        self.check_forward(cuda.cupy)
