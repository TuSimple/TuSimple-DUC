import ConfigParser
import os
import sys
import time
from PIL import Image

import cv2 as cv
import numpy as np

from tusimple_duc.test.tester import Tester


class ImageListTester:
    def __init__(self, config):
        self.config = config
        # # model
        self.model_dir = config.get('model', 'model_dir')
        self.model_prefix = config.get('model', 'model_prefix')
        self.model_epoch = config.getint('model', 'model_epoch')
        self.result_dir = config.get('model', 'result_dir')
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
        if not os.path.isdir(os.path.join(self.result_dir, 'visualization')):
            os.mkdir(os.path.join(self.result_dir, 'visualization'))
        if not os.path.isdir(os.path.join(self.result_dir, 'score')):
            os.mkdir(os.path.join(self.result_dir, 'score'))

        # data
        self.image_list = config.get('data', 'image_list')
        self.test_img_dir = config.get('data', 'test_img_dir')
        self.result_shape = [int(f) for f in config.get('data', 'result_shape').split(',')]
        # initialize tester
        self.tester = Tester(self.config)

    def predict_single(self, item):
        # img_name = item.strip().replace('/', '_')
        img_name = item.strip().split('/')[-1]
        img_path = os.path.join(self.test_img_dir, item.strip().split('\t')[1])

        # read image as rgb
        im = cv.imread(img_path)[:, :, ::-1]
        result_width = self.result_shape[1]
        result_height = self.result_shape[0]

        concat_img = Image.new('RGB', (result_width * 2, result_height * 2))

        results = self.tester.predict_single(
            img=im,
            ret_converted=True,
            ret_heat_map=True,
            ret_softmax=True)

        # label
        heat_map = results['heat_map']
        cvt_labels = results['converted']
        raw_labels = results['raw']
        softmax = results['softmax']

        confidence = float(np.max(softmax, axis=0).mean())

        result_img = Image.fromarray(self.tester.colorize(raw_labels)).resize(self.result_shape[::-1])

        # paste raw image
        concat_img.paste(Image.fromarray(im).convert('RGB'), (0, 0))
        # paste color result
        concat_img.paste(result_img, (0, result_height))
        # paste blended result
        concat_img.paste(Image.fromarray(cv.addWeighted(im[:, :, ::-1], 0.5, np.array(result_img), 0.5, 0)),
                         (result_width, 0))
        # paste heat map
        concat_img.paste(Image.fromarray(heat_map[:, :, [2, 1, 0]]).resize(self.result_shape[::-1]),
                         (result_width, result_height))
        concat_img.save(os.path.join(self.result_dir, 'visualization', img_name.replace('jpg', 'png')))

        # save results for score
        cv.imwrite(os.path.join(self.result_dir, 'score', img_name.replace('jpg', 'png')), cvt_labels)
        return confidence, img_path

    def predict_all(self):
        img_list = [line for line in open(self.image_list, 'r')]
        idx = 0
        conf_lst = []
        for item in img_list[:]:
            idx += 1
            start_time = time.time()
            conf_lst.append(self.predict_single(item))
            print 'Process %d out of %d image ... %s, time cost:%.3f, confidence:%.3f' % \
                  (idx, len(img_list), item.strip().split('/')[-1], time.time() - start_time, conf_lst[-1][0])
        conf_file = open(os.path.join(self.result_dir, self.model_prefix + str(self.model_epoch) + '.txt'), 'w')
        conf_lst.sort()
        for item in conf_lst:

            print >> conf_file, "{}\t{}".format(item[1], item[0])


if __name__ == '__main__':
    config_path = sys.argv[1]
    config = ConfigParser.RawConfigParser()
    config.read(config_path)
    tester = ImageListTester(config)
    tester.predict_all()
