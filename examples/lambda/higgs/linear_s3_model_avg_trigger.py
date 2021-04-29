import boto3
import json

from sync.sync_grad import clear_bucket
from storage.storage import s3_operator


def handler(event, context):

    # dataset setting
    dataset_name = 'higgs'
    bucket_name = "higgs-10"
    n_features = 30
    n_classes = 2
    tmp_bucket = "tmp-params"
    merged_bucket = "merged-params"

    # training setting
    algo = "lr"    # lr or svm
    sync_mode = "reduce"    # async, reduce or reduce_scatter
    num_workers = 10

    # hyper-parameters
    lr = 0.01
    batch_size = 10000
    n_epochs = 40
    valid_ratio = .2

    # clear s3 bucket
    s3_client = s3_operator.get_client()
    s3_operator.clear_bucket(s3_client, tmp_bucket)
    s3_operator.clear_bucket(s3_client, merged_bucket)

    # lambda payload
    payload = dict()
    payload['dataset'] = dataset_name
    payload['bucket_name'] = bucket_name
    payload['n_features'] = n_features
    payload['n_classes'] = n_classes
    payload['n_workers'] = num_workers
    payload['tmp_bucket'] = tmp_bucket
    payload['merged_bucket'] = merged_bucket
    payload['algo'] = algo
    payload['sync_mode'] = algo
    payload['lr'] = lr
    payload['batch_size'] = batch_size
    payload['n_epochs'] = n_epochs
    payload['valid_ratio'] = valid_ratio

    function_name = \
        "linear_s3_grad_avg_reduce" if sync_mode == "reduce" \
            else "linear_s3_grad_avg_reduce_scatter" if sync_mode == "reduce_scatter" \
            else "linear_s3_grad_avg_async"

    # invoke functions
    lambda_client = boto3.client('lambda')
    for i in range(num_workers):
        payload['rank'] = i
        payload['file'] = '{}_{}'.format(i, num_workers)
        lambda_client.invoke(FunctionName=function_name,
                             InvocationType='Event',
                             Payload=json.dumps(payload))
