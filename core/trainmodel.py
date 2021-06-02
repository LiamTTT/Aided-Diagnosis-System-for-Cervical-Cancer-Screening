from json import load
import tensorflow as tf
from keras import optimizers
from keras.utils import multi_gpu_model
from keras.callbacks import LearningRateScheduler

from .networks import create_model, load_weight, CREATE_CONFIG
from .lr_scheduler import Scheduler

TASK_METRICS = {
    "Classify": {
        "loss": ["binary_crossentropy"],
        "metrics": ["binary_accuracy"]
    },
    "Locate": {
        "loss": ["categorical_crossentropy"],
        "metrics": ["categorical_accuracy"]
    }

}

class TrainModel:
    """Compile model for training.
    """
    def __init__(self, cfg):
        print("Setting training model ...")
        self.CFG = cfg
        self.create_cfg = cfg["Model"]["model"]
        self.init_weight = cfg["Model"]["initial_weight"]
        self.locating = cfg["Model"]["locating_model"]
        self.nb_gpu = cfg["Opt"]["gpu_id"].split(',')
        self.task_metric = TASK_METRICS[self.create_cfg["task_type"]]
        # parse model create cfg
        if isinstance(self.create_cfg, str) and self.create_cfg in CREATE_CONFIG.keys():
            print("use predefined model creating config: {}".format(self.create_cfg))
            self.create_cfg = CREATE_CONFIG[cfg["Model"]["model"]]
        self.optimizer = optimizers.Adam(lr=cfg['hyp']['lr']) if not cfg['Opt']['SGD'] else optimizers.SGD(lr=cfg['hyp']['lr'])
        # create and compile model
        self.model = self._compile()
        
    def _compile(self):
        if len(self.nb_gpu)>1:
            print("multi-gpus training.")
            return self.__compile_muti_gpus()
        else:
            return self.__compile_single_gpu()
    
    def __compile_single_gpu(self):
        model = create_model(**self.create_cfg)
        model.compile(optimizer=self.optimizer, **self.task_metric)
        load_weight(model, self.init_weight, self.locating)
        return model
    
    def __compile_muti_gpus(self):
        with tf.device('/cpu:0'):
            model = self.__compile_single_gpu()
        parallel_model = multi_gpu_model(model, gpus=self.nb_gpu)
        parallel_model.compile(optimizer=self.optimizer, **self.task_metric)
        return parallel_model

        