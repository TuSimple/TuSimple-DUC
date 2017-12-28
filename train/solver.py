import os
import logging
from datetime import datetime

import mxnet as mx
from mxnet import metric
from tusimple_duc.core import utils
from tusimple_duc.core import lr_scheduler
from tusimple_duc.core.cityscapes_loader import CityLoader
from tusimple_duc.core.metrics import CompositeEvalMetric, AccWithIgnoreMetric, IoUMetric, SoftmaxLoss
from tusimple_duc.networks.network_duc_hdc import get_symbol_duc_hdc


class Solver(object):
    def __init__(self, config):
        try:
            self.config = config
            # environment
            self.use_cpu = config.getboolean('env', 'use_cpu')
            self.gpus = config.get('env', 'gpus')
            self.kv_store = config.get('env', 'kv_store')
            self.ctx = mx.cpu() if self.use_cpu is True else [
                mx.gpu(int(i)) for i in self.gpus.split(',')]
            self.multi_thread = config.getboolean('env', 'multi_thread')

            # network parameter
            self.label_num = config.getint('network', 'label_num')
            self.aspp = config.getint('network', 'aspp')
            self.aspp_stride = config.getint('network', 'aspp_stride')
            self.cell_width = config.getint('network', 'cell_width')
            self.ignore_label = config.getint('network', 'ignore_label')
            self.bn_use_global_stats = config.getboolean('network', 'bn_use_global_stats')

            # model
            self.num_epochs = config.getint('model', 'num_epochs')
            self.model_dir = config.get('model', 'model_dir')
            self.save_model_prefix = config.get('model', 'save_model_prefix')
            self.checkpoint_interval = config.getint('model', 'checkpoint_interval')
            # SGD parameters
            self.optimizer = 'sgd'
            self.lr = config.getfloat('model', 'lr')
            self.lr_policy = config.get('model', 'lr_policy')
            self.lr_factor = config.getfloat('model', 'lr_factor')
            self.lr_factor_epoch = config.getfloat('model', 'lr_factor_epoch')
            self.momentum = config.getfloat('model', 'momentum')
            self.weight_decay = config.getfloat('model', 'weight_decay')
            # fine tuning
            self.load_model_dir = config.get('model', 'load_model_dir')
            self.load_model_prefix = config.get('model', 'load_model_prefix')
            self.load_epoch = config.getint('model', 'load_epoch')
            # evaluation metric
            self.eval_metric = [m.strip() for m in config.get('model', 'eval_metric').split(',')]

            # data
            self.data_dir = config.get('data', 'data_dir')
            self.label_dir = config.get('data', 'label_dir')
            self.train_list = config.get('data', 'train_list')
            self.use_val = config.getboolean('data', 'use_val')
            if self.use_val:
                self.val_list = config.get('data', 'val_list')
            self.rgb_mean = tuple([float(color.strip()) for color in config.get('data', 'rgb_mean').split(',')])
            self.batch_size = config.getint('data', 'batch_size')
            self.ds_rate = config.getint('data', 'ds_rate')
            self.convert_label = config.getboolean('data', 'convert_label')
            self.scale_factors = [float(scale.strip()) for scale in config.get('data', 'scale_factors').split(',')]
            self.crop_shape = tuple([int(l.strip()) for l in config.get('data', 'crop_shape').split(',')])
            self.use_mirror = config.getboolean('data', 'use_mirror')
            self.use_random_crop = config.getboolean('data', 'use_random_crop')
            self.random_bound = tuple([int(l.strip()) for l in config.get('data', 'random_bound').split(',')])

            # miscellaneous
            self.draw_network = config.getboolean('misc', 'draw_network')

            # inference
            self.train_size = 0
            with open(self.train_list, 'r') as f:
                for _ in f:
                    self.train_size += 1
            self.epoch_size = self.train_size / self.batch_size
            self.data_shape = [tuple(list([self.batch_size, 3, self.crop_shape[0], self.crop_shape[1]]))]
            self.label_shape = [tuple([self.batch_size, (self.crop_shape[1]*self.crop_shape[0]/self.cell_width**2)])]
            self.data_name = ['data']
            self.label_name = ['seg_loss_label']
            self.symbol = None
            self.arg_params = None
            self.aux_params = None

        except ValueError:
            logging.error('Config parameter error')

    def get_data_iterator(self):
        loader = CityLoader
        train_args = {
            'data_path'             : self.data_dir,
            'label_path'            : self.label_dir,
            'rgb_mean'              : self.rgb_mean,
            'batch_size'            : self.batch_size,
            'scale_factors'         : self.scale_factors,
            'data_name'             : self.data_name,
            'label_name'            : self.label_name,
            'data_shape'            : self.data_shape,
            'label_shape'           : self.label_shape,
            'use_random_crop'       : self.use_random_crop,
            'use_mirror'            : self.use_mirror,
            'ds_rate'               : self.ds_rate,
            'convert_label'         : self.convert_label,
            'multi_thread'          : self.multi_thread,
            'cell_width'            : self.cell_width,
            'random_bound'          : self.random_bound,
        }
        val_args = train_args.copy()
        val_args['scale_factors'] = [1]
        val_args['use_random_crop'] = False
        val_args['use_mirror'] = False
        train_dataloader = loader(self.train_list, train_args)
        if self.use_val:
            val_dataloader = loader(self.val_list, val_args)
        else:
            val_dataloader = None
        return train_dataloader, val_dataloader

    def get_symbol(self):
        self.symbol = get_symbol_duc_hdc(
            cell_cap=(self.ds_rate / self.cell_width) ** 2,
            label_num=self.label_num,
            ignore_label=self.ignore_label,
            bn_use_global_stats=self.bn_use_global_stats,
            aspp_num=self.aspp,
            aspp_stride=self.aspp_stride,
        )

    # build up symbol, parameters and auxiliary parameters
    def get_model(self):
        self.get_symbol()

        # load model
        if self.load_model_prefix is not None and self.load_epoch > 0:
            self.symbol, self.arg_params, self.aux_params = \
                mx.model.load_checkpoint(os.path.join(self.load_model_dir, self.load_model_prefix), self.load_epoch)

    def fit(self):
        # kvstore
        if self.kv_store is 'local' and (
                self.gpus is None or len(self.gpus.split(',')) is 1):
            kv = None
        else:
            kv = mx.kvstore.create(self.kv_store)

        # setup module, including symbol, params and aux
        # get_model should always be called before get_data_iterator to ensure correct data loader
        self.get_model()

        # get dataloader
        train_data, eval_data = self.get_data_iterator()

        # evaluate metrics
        eval_metric_lst = []
        if "acc" in self.eval_metric:
            eval_metric_lst.append(metric.create(self.eval_metric))
        if "acc_ignore" in self.eval_metric and self.ignore_label is not None:
            eval_metric_lst.append(AccWithIgnoreMetric(self.ignore_label, name="acc_ignore"))
        if "IoU" in self.eval_metric and self.ignore_label is not None:
            eval_metric_lst.append(IoUMetric(self.ignore_label, label_num=self.label_num, name="IoU"))
        eval_metric_lst.append(SoftmaxLoss(self.ignore_label, label_num=self.label_num, name="SoftmaxLoss"))
        eval_metrics = CompositeEvalMetric(metrics=eval_metric_lst)

        optimizer_params = {}
        # optimizer
        # lr policy
        if self.lr_policy == 'step' and self.lr_factor < 1 and self.lr_factor_epoch > 0:
            optimizer_params['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
                step=max(int(self.epoch_size * self.lr_factor_epoch), 1),
                factor=self.lr_factor)
        elif self.lr_policy == 'poly':
            optimizer_params['lr_scheduler'] = lr_scheduler.PolyScheduler(
                origin_lr=self.lr,
                max_samples=max(int(self.epoch_size * self.num_epochs), 1),
                factor=self.lr_factor)
        else:
            logging.error('Unknown lr policy: %s' % self.lr_policy)
        optimizer_params['learning_rate'] = self.lr
        optimizer_params['momentum'] = self.momentum
        optimizer_params['wd'] = self.weight_decay
        optimizer_params['rescale_grad'] = 1.0 / self.batch_size
        optimizer_params['clip_gradient'] = 5

        # directory for saving models
        model_path = os.path.join(self.model_dir, self.save_model_prefix)
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        model_full_path = os.path.join(model_path, datetime.now().strftime('%Y_%m_%d_%H:%M:%S'))
        if not os.path.isdir(model_full_path):
            os.mkdir(model_full_path)
        checkpoint = utils.do_checkpoint(os.path.join(model_full_path, self.save_model_prefix), self.checkpoint_interval)
        with open(os.path.join(model_full_path,
                               'train_' + datetime.now().strftime('%Y_%m_%d_%H:%M:%S') + '.cfg'), 'w') as f:
            self.config.write(f)
        utils.save_symbol(self.symbol, os.path.join(model_full_path, self.save_model_prefix))
        utils.save_log(self.save_model_prefix, model_full_path)

        # draw network
        if self.draw_network is True:
            utils.draw_network(self.symbol, os.path.join(model_full_path, self.save_model_prefix), self.data_shape[0])

        # batch_end_callback
        batch_end_callback = list()
        batch_end_callback.append(utils.Speedometer(self.batch_size, 10))

        module = mx.module.Module(self.symbol, context=self.ctx, data_names=self.data_name, label_names=self.label_name)

        # initialize (base_module now no more do this initialization)
        train_data.reset()
        module.fit(
            train_data=train_data,
            eval_data=eval_data,
            eval_metric=eval_metrics,
            epoch_end_callback=checkpoint,
            batch_end_callback=batch_end_callback,
            kvstore=kv,
            optimizer=self.optimizer,
            optimizer_params=optimizer_params,
            initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
            arg_params=self.arg_params,
            aux_params=self.aux_params,
            allow_missing=True,
            begin_epoch=self.load_epoch,
            num_epoch=self.num_epochs,
        )
