import os
import numpy as np
from s3.get_object import get_object
from utils.file_utils import split_file_with_info2
from sync.sync_grad_redis import clear_bucket
from elasticache.Redis.__init__ import redis_init
from elasticache.Redis.set_object import hset_object
from elasticache.Redis.get_object import hget_object

num_files = 20

redis_location = "test.fifamc.ng.0001.euc1.cache.amazonaws.com"
grad_bucket = "tmp-grads"
model_bucket = "tmp-updates"
endpoint = redis_init(redis_location)
def main(event,context):
    # clear everything before start  
    clear_bucket(endpoint, model_bucket)
    clear_bucket(endpoint,grad_bucket)
    hset_object(endpoint, model_bucket, "counter", 0)
    print("double check for synchronized flag = {}".format(hget_object(endpoint, model_bucket, "counter")))
    src_dir = "agaricus"
    src_file = get_object(src_dir,"agaricus_127d_train.libsvm").read().decode('utf-8').split("\n")#return 'body':bytestream
    dst_dir = "agarics-libsvm"
    split_file_with_info2(src_file, dst_dir, num_files)

    

