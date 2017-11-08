import logging
import numpy as np
import mxnet as mx
from mxnet.metric import EvalMetric


class CompositeEvalMetric(EvalMetric):
    """Manage multiple evaluation metrics."""

    def __init__(self, **kwargs):
        super(CompositeEvalMetric, self).__init__('composite')
        try:
            self.metrics = kwargs['metrics']
        except KeyError:
            self.metrics = []

    def add(self, metric):
        self.metrics.append(metric)

    def get_metric(self, index):
        try:
            return self.metrics[index]
        except IndexError:
            return ValueError("Metric index {} is out of range 0 and {}".format(
                index, len(self.metrics)))

    def update(self, labels, preds):
        for metric in self.metrics:
            metric.update(labels, preds)

    def reset(self):
        try:
            for metric in self.metrics:
                metric.reset()
        except AttributeError:
            pass

    def get(self):
        names = []
        results = []
        for metric in self.metrics:
            result = metric.get()
            names.append(result[0])
            results.append(result[1])
        return names, results

    def print_log(self):
        names, results = self.get()
        logging.info('; '.join(['{}: {}'.format(name, val) for name, val in zip(names, results)]))


def check_label_shapes(labels, preds, shape=0):
    if shape == 0:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))


class AccWithIgnoreMetric(EvalMetric):
    def __init__(self, ignore_label, name='AccWithIgnore'):
        super(AccWithIgnoreMetric, self).__init__(name=name)
        self._ignore_label = ignore_label
        self._iter_size = 200
        self._nomin_buffer = []
        self._denom_buffer = []

    def update(self, labels, preds):
        check_label_shapes(labels, preds)
        for i in range(len(labels)):
            pred_label = mx.ndarray.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')

            check_label_shapes(label, pred_label)

            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat) - (label.flat == self._ignore_label).sum()


class IoUMetric(EvalMetric):
    def __init__(self, ignore_label, label_num, name='IoU'):
        self._ignore_label = ignore_label
        self._label_num = label_num
        super(IoUMetric, self).__init__(name=name)

    def reset(self):
        self._tp = [0.0] * self._label_num
        self._denom = [0.0] * self._label_num

    def update(self, labels, preds):
        check_label_shapes(labels, preds)
        for i in range(len(labels)):
            pred_label = mx.ndarray.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')

            check_label_shapes(label, pred_label)

            iou = 0
            eps = 1e-6
            # skip_label_num = 0
            for j in range(self._label_num):
                pred_cur = (pred_label.flat == j)
                gt_cur = (label.flat == j)
                tp = np.logical_and(pred_cur, gt_cur).sum()
                denom = np.logical_or(pred_cur, gt_cur).sum() - np.logical_and(pred_cur, label.flat == self._ignore_label).sum()
                assert tp <= denom
                self._tp[j] += tp
                self._denom[j] += denom
                iou += self._tp[j] / (self._denom[j] + eps)
            iou /= self._label_num
            self.sum_metric = iou
            self.num_inst = 1


class SoftmaxLoss(EvalMetric):
    def __init__(self, ignore_label, label_num, name='OverallSoftmaxLoss'):
        super(SoftmaxLoss, self).__init__(name=name)
        self._ignore_label = ignore_label
        self._label_num = label_num

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        loss = 0.0
        cnt = 0.0
        eps = 1e-6
        for i in range(len(labels)):
            prediction = preds[i].asnumpy()[:]
            shape = prediction.shape
            if len(shape) == 4:
                shape = (shape[0], shape[1], shape[2]*shape[3])
                prediction = prediction.reshape(shape)
            label = labels[i].asnumpy()
            soft_label = np.zeros(prediction.shape)
            for b in range(soft_label.shape[0]):
                for c in range(self._label_num):
                    soft_label[b][c][label[b] == c] = 1.0

            loss += (-np.log(prediction[soft_label == 1] + eps)).sum()
            cnt += prediction[soft_label == 1].size
        self.sum_metric += loss
        self.num_inst += cnt

