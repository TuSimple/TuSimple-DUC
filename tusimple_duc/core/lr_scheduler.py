"""
poly learning rate scheduler, re-implement the 'poly' learning policy from caffe.
learning rate decays as epochs grows
"""
from mxnet.lr_scheduler import LRScheduler
import logging


class PolyScheduler(LRScheduler):
    """Reduce learning rate in a power way
    Assume the weight has been updated by n times, then the learning rate will
    be
    base_lr * (floor(1-n/max_time))^factor
    Parameters
    ----------
    origin_lr: int
        original learning rate
    max_samples: int
        schedule learning rate after n updates
    show_num: int
        show current learning rate after n updates
    factor: float
        the factor for reducing the learning rate
    """
    def __init__(self, origin_lr, max_samples, show_num=10,factor=1, stop_factor_lr=1e-8):
        super(PolyScheduler, self).__init__()
        if max_samples < 1:
            raise ValueError("Schedule max time must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.max_samples = max_samples
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.count = 0
        self.origin_lr = origin_lr
        self.base_lr = origin_lr
        self.show_num = show_num

    def __call__(self, num_update):
        """
        Call to schedule current learning rate
        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """
        if num_update > self.count:
            self.base_lr = self.origin_lr * pow((1 - 1.0*num_update/self.max_samples), self.factor)
            if self.base_lr < self.stop_factor_lr:
                self.base_lr = self.stop_factor_lr
                logging.info("Update[%d]: now learning rate arrived at %0.5e, will not "
                             "change in the future", num_update, self.base_lr)
            elif num_update % self.show_num == 0:
                logging.info("Update[%d]: Change learning rate to %0.8e",
                             num_update, self.base_lr)
                self.count = num_update
        return self.base_lr
