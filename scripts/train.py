import os
import sys
import yaml
import argparse

from core.networks import create_model, CREATE_CONFIG
from core.trainmodel import TrainModel
from core.dataset import Dataset


def create_save(project_root, model_alias):
    train_dir = os.path.join(project_root, model_alias)
    cfg_dir = os.path.join(train_dir, "configs")
    ckpt_dir = os.path.join(train_dir, "weights")
    tb_dir = os.path.join(train_dir, "tb_log")
    log_path = os.path.join(train_dir, "train.log")
    print("Project save at: {}".format(project_root))
    print("Training recording at {}".format(train_dir))
    os.makedirs(project_root, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    return project_root, train_dir, cfg_dir, ckpt_dir, tb_dir, log_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./data_for_test/configs/train_model1_cls.yaml")
    Args = parser.parse_args()
    
    cfg = yaml.load(open(Args.config, 'r'))

    os.environ["CUDA_VISIBLE_DEVICES"]=cfg["Opt"]["gpu_id"]
    
    # save setting
    project_root, train_dir, \
    cfg_dir, ckpt_dir, tb_dir, \
    log_path = create_save(cfg["Project"], cfg["Model"]["alias"])
    
    # model setting
    train_model = TrainModel(cfg)
    
    # train setting 
    start_epoch = cfg["Opt"]["start_epoch"]
    end_epochs = cfg["Opt"]["end_epoch"]
    