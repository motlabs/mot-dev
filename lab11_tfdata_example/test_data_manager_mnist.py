import os
from glob import glob
from PIL import Imag
import numpy as np


dataset_home_path = '/Users/jwkangmacpro/SourceCodes/datamanager_example/mnist_png/'
datatype = 'training/'
classtype = '*/'

datapath = dataset_home_path +  datatype + classtype + '*.png'
data_list = glob(datapath)

path = data_list[0]
path.split('/')[-2]


def get_label_from_path(path):
    return path.split('/')[-2]

# we can obtain the label and path of the dataset
path, get_label_from_path(path)


rand_n = 9999

path = data_list[rand_n]
path, get_label_from_path(path)