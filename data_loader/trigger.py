import os
import numpy as np
from s3.get_object import get_object
from utils.file_utils import split_file_with_info2

num_files = 5

def main(event,context):
    src_dir = "agaricus"
    src_fil e= get_object(src_dir,"agaricus_127d_train.libsvm").read().decode('utf-8').split("\n")#return 'body':bytestream
    dst_dir = "agarics-libsvm"
    split_file_with_info(src_file, dst_dir, num_files)

