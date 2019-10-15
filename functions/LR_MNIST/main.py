from dataset_utils import preprocess, partition_training_data
import os

def handler(event,context):
	
	dataset_name = 'mnist'
	bucket_name = 'lr.mnist'
	training_keys = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz']
	test_keys = ['t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
	num_workers = 4

	data_path = os.path.join(os.sep, 'tmp', 'data', dataset_name)

	# return processed training samples and labels for following partition; save processed test samples and labels back to S3
	training_x, training_y = preprocess(bucket_name, training_keys, test_keys, data_path)

    # shuffle and partition training data
	partition_training_data(bucket_name, data_path, training_x, training_y, num_workers)

