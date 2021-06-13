from multiprocessing.spawn import import_main_path
import os
import difflib
import time
import imghdr
import pickle
import cv2
import multiprocessing as mp
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from glob2 import glob
from core.augmentations import Augmentations


class Sample:
    
    def __init__(self, img_path, label, src_mpp, src_size, mask_path=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.label = label
        self.src_mpp = src_mpp
        self.src_size = src_size
        # self.dst_mpp = dst_mpp
        # self.dst_size = dst_size
        # # todo where the annotation come from (Dev Batch Fold Coordinate Subclass etc.)
        # # for current training
        # self.src_crop = int(1.5*dst_size*dst_mpp/src_mpp)
        # self.dst_crop = int(1.5*dst_size)
        
    def __repr__(self):
        return "<{} object at {}> Label: {}, SrcMpp: {}, SrcSize: {}, ImgPath: {}, MaskPath: {}".format(
            type(self), hex(id(self)), 
            self.label, self.src_mpp, self.src_size, self.img_path, self.mask_path
            )

        
class Subset:
    
    def __init__(self, pool, sampling_nb, gain, mask_flg=False, remark=None):
        self.pool = pool;
        self.sampling_nb = sampling_nb
        self.gain = gain
        self.remark = remark
        self.mask_flg = mask_flg
        # 
        self.final_sampling = min(len(pool), int(gain*sampling_nb))
        
    def __len__(self):
        return len(self.pool)
    
    def __repr__(self):
        return "<{} object at {}> Remark: {}, Mask: {}, Pool: {}, Sampling: {}, Gain: {}, Final: {}".format(
            type(self), hex(id(self)), 
            self.remark, self.mask_flg, len(self), self.sampling_nb, self.gain, self.final_sampling
            )


class Dataset:
    
    def __init__(self, config):
        # todo copy the dataset file to train dir.
        
        pass
    
    def gen_train(self):
        pass


# =========== function for preprocess ==========
def validate_img(img_path):
    if img_path and os.path.isfile(img_path):
        return imghdr.what(img_path) != None
    return False


def preprocess(sample, dst_mpp, dst_size):
    pass

    
def _parse_dataset_file(path, sheet_name='train', mask_flg=False):
    sheet = pd.read_excel(path, sheet_name, 
                          usecols=['Flag', 'Samples', 'Masks', 'Remark', 'src_size', 'src_mpp', 'label', 'Gain', 'Num']
                          )
    sheet = sheet.loc[sheet['Flag']==1]  # filtering Flag==0
    parse_row = partial(_parse_row, mask_flg=mask_flg)
    with mp.Pool(6) as p:
        subsets = p.map(parse_row, sheet.iterrows())
    return subsets


def _parse_row(row, mask_flg=False):
    """parse single dataset.xlsx row`, return a Subset object.
    """
    # pool attr
    _, row = row
    samples_rt = row["Samples"]
    masks_rt = row["Masks"]
    # subset attr
    sampling_nb = row["Num"]
    gain = row["Gain"]
    remark = row["Remark"]
    # sample attr
    src_size = row["src_size"]
    src_mpp = row["src_mpp"]
    label = row["label"]
    
    samples_ls = __get_image_list(samples_rt)
    masks_ls = [None]*len(samples_ls) 
    if label==0:
        # for negative samples, '0' for generated empty mask
        masks_ls = [0]*len(samples_ls)
    if masks_rt == masks_rt:
        # get all mask pool and find the mask according to image name.
        _masks_pool = __get_image_list(masks_rt)
        _mask_names = [os.path.splitext(os.path.basename(p))[0] for p in _masks_pool]
        for idx, sample_path in enumerate(samples_ls):
            mask_name = matching_sample_mask(sample_path, _mask_names)
            masks_ls[idx] = mask_name if mask_name is None else _masks_pool[_mask_names.index(mask_name)]

    # collecting Sample object according to sample path in samples_ls
    # validating the image file
    start = time.time()
    valmask_flgs = [True]*len(samples_ls)
    with ThreadPoolExecutor(max_workers=30) as exc:
        valsample_flgs = exc.map(validate_img, samples_ls)
        if mask_flg and label!=0:
            valmask_flgs = exc.map(validate_img, masks_ls)
    val_flgs = valsample_flgs and valmask_flgs
    end = time.time()
    print("{} val time: {}".format(_, end-start))
    sample_pool = [Sample(img_path, label, src_mpp, src_size, mask_path) for img_path, mask_path, val_flg in zip(samples_ls, masks_ls, val_flgs) if val_flg]
    subset = Subset(sample_pool, sampling_nb, gain, mask_flg, remark)
    # # print message of current subset
    # print(subset)
    return subset
        


def __get_image_list(rt):
    if os.path.isfile(rt):
        # read from image list file, stripping invalid path item.
        img_ls = [l.strip('\n') for l in open(rt, 'r').readlines() if os.path.isfile(l.strip('\n'))]
    elif os.path.isdir(rt):
        # get from image dir directly.
        img_ls = glob(os.path.join(rt, "*.[j,p,t][p,n,i][g,f]")) # for jpg, png, tif
    else:
        raise ValueError("Check root of images: {}".format(rt))
    return img_ls


def matching_sample_mask(sample_path, mask_names):
    """search mask of sample according to image name
    """
    sample_name = os.path.splitext(os.path.basename(sample_path))[0]
    matching_ls = difflib.get_close_matches(sample_name, mask_names, cutoff=0.9)
    return matching_ls[0] if len(matching_ls) else None
    


if __name__ == '__main__':
    start = time.time()
    sheet = _parse_dataset_file(r'D:\liusibo\Codes\GitHub\Aided-Diagnosis-System-for-Cervical-Cancer-Screening\data_for_test\configs\dataset.xlsx', 'test', True)
    end = time.time()
    print('{:.3f} s'.format(end-start))
    print(sheet.to_dict('list'))
        