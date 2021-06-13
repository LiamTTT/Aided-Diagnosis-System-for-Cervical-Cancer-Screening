import os
from glob2 import glob

if __name__ == '__main__':
    pr = "./data_for_test/tiles/samples/A/test/p"
    nr = "./data_for_test/tiles/samples/A/test/n"
    
    p_ls = [f+'\n' for f in glob(os.path.join(pr, '*.tif'))]
    n_ls = [f+'\n' for f in glob(os.path.join(nr, '*.tif'))]
    
    with open('./data_for_test/tiles/samples/test_A_p.txt', 'w+') as f:
        f.writelines(p_ls)
    
    with open('./data_for_test/tiles/samples/test_A_n.txt', 'w+') as f:
        f.writelines(n_ls)
    