from .dataset_utils import preprocess_cifar10
import os
import boto3
import json
import time


def handler(event, context):
    dataset_name = 'cifar10'
    bucket_name = 'cifar10dataset'
    num_workers = 8

    # invoke functions
    payload = dict()
    payload['dataset'] = dataset_name
    payload['data_bucket'] = bucket_name
    # payload['model_bucket'] = 'model_bucket'
    payload['num_workers'] = num_workers

    # invoke functions
    lambda_client = boto3.client('lambda')
    for i in range(num_workers):
        payload['rank'] = i
        payload['keys_training_data'] = 'training_{}.pt'.format(i)
        payload['keys_test_data'] = 'test.pt'
        lambda_client.invoke(FunctionName='cifar10_F2', InvocationType='Event', Payload=json.dumps(payload))
