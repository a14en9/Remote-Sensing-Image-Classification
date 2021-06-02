import numpy as np
import os, sys
import random
import shutil
from sklearn.cross_validation import train_test_split

def random_copy_file(source_dir, train_target_dir, test_target_dir, train_size):
    samples = os.listdir(source_dir)
    idx = len(samples)
    range_idx = np.arange(idx)
    # include_index = np.random.choice(idx, int(train_size*idx))
    include_index = random.sample(list(range(idx)), int(train_size*idx))
    mask = np.zeros(range_idx.shape, dtype=bool)
    mask[include_index] = True
    exclude = range_idx[mask]
    include = range_idx[~mask]
    for i in range(len(include)):
        dir = os.path.join(source_dir, samples[include[i]])
        shutil.move(dir, train_target_dir)
    for n in range(len(exclude)):
        dirs = os.path.join(source_dir, samples[exclude[n]])
        shutil.move(dirs, test_target_dir)

if __name__ == '__main__':
    root_path = "/home/ailab/workspace/rssc/"
    train_path = os.path.join(root_path, "AID_05/train/")
    test_path = os.path.join(root_path, "AID_05/test/")
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    source_root = os.path.join(root_path, "datasets/AID/")
    dirs = os.listdir(path=source_root)
    for file in dirs:
        filename = os.path.join(source_root, file+'/')
        train_file = os.path.join(train_path,file+'/')
        test_file = os.path.join(test_path, file+'/')
        if not os.path.exists(train_file):
            os.makedirs(train_file)
        if not os.path.exists(test_file):
            os.makedirs(test_file)
        random_copy_file(filename, train_file, test_file,0.5)