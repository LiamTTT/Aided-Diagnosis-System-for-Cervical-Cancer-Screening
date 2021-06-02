import math
import keras.backend as K

_EPSILON = K.epsilon()
class Scheduler:
    @staticmethod
    def Cosine(epochs, init_lr):
        def cosine(epoch):
            lr = init_lr * ( math.cos(epoch*math.pi)/epochs + 1) / 2
            return lr if lr>_EPSILON else _EPSILON
        return cosine
    
    @staticmethod
    def Linear(epochs, init_lr):
        def linear(epoch):
            lr = init_lr*(1-epoch/epochs)
            return lr if lr>_EPSILON else _EPSILON
        return linear
    
    @staticmethod
    def Staging(epochs, init_lr, s=3):
        def staging(epoch):
            split = epochs//s
            lr = init_lr*(1 - (epoch//split)/3)
            return lr if lr>_EPSILON else _EPSILON
        return staging
        
    # todo more lr scheduler strategy

SCHDULERS = {
    "Cosine": Scheduler.Cosine,
    "Linear": Scheduler.Linear,
    "Stage": Scheduler.Staging
}