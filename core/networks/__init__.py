from keras.backend.cntk_backend import ndim
from .BasingResNet50 import *
from .BasingRNN import *

__all__ = ["create_model", "CREATE_CONFIG"]

VALID_BACKBONES = ["ResNet50", "RNN"]
VALID_TASKS = ["Classify", "Locate", "Model1", "Model2"]


def create_model(backbone, task_type, input_shape, esembly=False,**kargs):
    assert backbone in VALID_BACKBONES, "Chosssing backbone from {}".format(VALID_BACKBONES)
    assert task_type in VALID_TASKS, "Chosssing task from {}".format(VALID_TASKS)

    if backbone == "ResNet50":
        # for final model esembling
        if esembly:
            if task_type == "Model1":
                print("Create Model1")
                return Resnet50_model1(input_shape=input_shape)
            elif task_type == "Model2":
                print("Create Model2")
                return ResNet50_model2(input_shape=input_shape)
            else:
                raise ValueError("Task type must be Model 1 or 2 when create esembly model.")
        # for training 
        if task_type == "Classify":
            print("Create Classifing Model\ninput shape: {}".format(input_shape))
            return ResNet50_classify(input_shape=input_shape)
        elif task_type == "Locate":
            print("Create Locating Model\ninput shape: {}".format(input_shape))
            return Resnet50_locate(input_shape=input_shape)
        else:
            raise ValueError("Task type could not be Model 1 or 2 when training.")
    elif backbone == "RNN":
        if task_type == "Classify":
            return RNN_classify(input_shape=input_shape, ndim=ndim)
        else:
            raise ValueError("Classify only when backbone is RNN")
        

CREATE_CONFIG = {
    "Model1":
        {
            "backbone": "ResNet50",
            "task_type": "Model1",
            "input_shape": (512, 512, 3),
            "esembly": True
        },
    "Model2":
        {
            "backbone": "ResNet50",
            "task_type": "Model2",
            "input_shape": (256, 256, 3),
            "esembly": True
        },
    "TrModel1Clf":
        {
            "backbone": "ResNet50",
            "task_type": "Classify",
            "input_shape": (512, 512, 3)
        },
    "TrModel1Loc":
        {
            "backbone": "ResNet50",
            "task_type": "Locate",
            "input_shape": (512, 512, 3)
        },
    "TrModel2Clf":
        {
            "backbone": "ResNet50",
            "task_type": "Classify",
            "input_shape": (256, 256, 3)
        },
    "RNN10":
        {
            "backbone": "RNN",
            "task_type": "Classify",
            "input_shape": (10, 2048),
            "ndim": 512
        },
    "RNN20":
        {
            "backbone": "RNN",
            "task_type": "Classify",
            "input_shape": (20, 2048),
            "ndim": 512
        },
    "RNN30":
        {
            "backbone": "RNN",
            "task_type": "Classify",
            "input_shape": (30, 2048),
            "ndim": 512
        }
}    