import mxnet as mx
import cv2 as cv
import numpy as np


class Predictor:
    def __init__(self, modules, label_num, ds_rate=8, cell_width=2,
                 result_shape=(1024, 2048), test_scales=[1]):
        self._modules = modules
        self._label_num = label_num

        self._ds_rate = ds_rate
        self._cell_width = cell_width
        self._rpn_width = self._ds_rate / self._cell_width
        self._result_shape = result_shape
        self._test_scales = test_scales
        self._im_shape = None

    def predict(self, imgs):
        result_height, result_width = self._result_shape
        label_list = []

        # multi scale test
        for index, test_scale in enumerate(self._test_scales):
            _, _, img_height, img_width = imgs[index].shape

            class CustomNDArrayIter(mx.io.NDArrayIter):
                @property
                def provide_data(self):
                    return [('data', self.data.shape)]
            data_iter = CustomNDArrayIter(imgs[index], np.zeros(1), 1, shuffle=False)
            labels = self._modules[index].predict(data_iter).asnumpy().squeeze()
            test_width = (int(img_width) / self._ds_rate) * self._ds_rate
            test_height = (int(img_height) / self._ds_rate) * self._ds_rate
            feat_width = test_width / self._ds_rate
            feat_height = test_height / self._ds_rate
            # re-arrange duc results
            labels = labels.reshape((self._label_num, self._ds_rate/self._cell_width, self._ds_rate/self._cell_width,
                                     feat_height, feat_width))
            labels = np.transpose(labels, (0, 3, 1, 4, 2))
            labels = labels.reshape((self._label_num, test_height / self._cell_width, test_width / self._cell_width))

            labels = labels[:, :int(img_height / self._cell_width),
                               :int(img_width / self._cell_width)]
            labels = np.transpose(labels, [1, 2, 0])
            labels = cv.resize(labels, (result_width, result_height), interpolation=cv.INTER_LINEAR)
            labels = np.transpose(labels, [2, 0, 1])
            label_list.append(labels)
        labels = np.array(label_list).sum(axis=0) / len(label_list)

        return labels
