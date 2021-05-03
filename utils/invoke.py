import boto3
import json

from sync.sync_centroids import clear_bucket

nr_files = 50
client = boto3.client('lambda')

payload = dict()
payload['bucket_name'] = "s3-libsvm"
payload['max_dim'] = 29
payload['num_clusters'] = 10000
payload['worker_cent_bucket'] = "worker-centroids"
payload['avg_cent_bucket'] = "avg-centroids"
payload['num_epochs'] = 15
payload['threshold'] = 0.02
clear_bucket(payload['worker_cent_bucket'])
clear_bucket(payload['avg_cent_bucket'])

for i in range(nr_files):
    payload['key'] = f"{i}_{nr_files}"
    response = client.invoke(
        FunctionName='test-kmeans',
        InvocationType='Event',
        Payload=json.dumps(payload))
