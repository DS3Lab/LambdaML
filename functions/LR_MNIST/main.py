from dataset_utils import preprocess_mnist, shuffle_and_partition
import os
import boto3
import torch
import json
import time

def handler(event,context):
	
	dataset_name = 'mnist'
	bucket_name = 'lr.mnist'
	training_keys = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz']
	test_keys = ['t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
	num_workers = 4
	
	data_path = os.path.join(os.sep, 'tmp', 'data', dataset_name)

	# return processed training samples and labels for following partition; save processed test samples and labels back to S3
	preprocess_start = time.time()
	training_x, training_y = preprocess_mnist(bucket_name, training_keys, test_keys, data_path)
	print("pre-process mnist cost {} s".format(time.time()-preprocess_start))
    
    # shuffle and partition training data
	partition_start = time.time()
	shuffle_and_partition(bucket_name, data_path, training_x, training_y, num_workers)
	print("partition dataset cost {} s".format(time.time()-partition_start))

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
		lambda_client.invoke(FunctionName='mnist', InvocationType='Event', Payload=json.dumps(payload))
		

