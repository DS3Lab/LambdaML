import os
import json
import urllib.parse
import boto3
import logging
import numpy as np

from s3.list_objects import list_bucket_objects
from s3.get_object import get_object

from model.LogisticRegression import LogisticRegression

tmp_bucket = "tmp-updates"
num_workers = 2
model_bucket = ""


def handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')

    print('bucket = {}'.format(bucket))
    print('key = {}'.format(key))

    # download file
    file = get_object(bucket, key).read().decode('utf-8')
    print(file)

    # Set up logging
    # logging.basicConfig(level=logging.INFO, format='%(message)s')

    sum_w = np.zeros([2, 3])
    tmp_path = weight_path = '/tmp/'

    s_3 = boto3.client('s3')

    # Retrieve the bucket's objects
    # objects = list_bucket_objects(tmp_bucket)
    # if objects is not None:
    #     # List the object names
    #     print('Objects in {}'.format(tmp_bucket))
    #     for obj in objects:
    #         file_key = urllib.parse.unquote_plus(obj["Key"], encoding='utf-8')
    #         print('file:  {}'.format(file_key))
    #         s_3.download_file(tmp_bucket, file_key, tmp_path + str(file_key))
    #         w = np.loadtxt(tmp_path + str(file_key))
    #         sum_w = sum_w + w
    #     print(sum_w)
    # else:
    #     # Didn't get any keys
    #     print('No objects in {}'.format(tmp_bucket))


if __name__ == '__main__':

    bucket = "agaricus"
    key = "agaricus_127d_train.libsvm"

    print('bucket = {}'.format(bucket))
    print('key = {}'.format(key))

    objects = list_bucket_objects(bucket)
    if objects is not None:
        # List the object names
        logging.info(f'Objects in {bucket}')
        for obj in objects:
            logging.info(f'  {obj["Key"]}')
    else:
        # Didn't get any keys
        logging.info(f'No objects in {bucket}')

    # download file
    file = get_object(bucket, key)
    body = file.read().decode('utf-8')
    print("body of file:")
    print(file)
