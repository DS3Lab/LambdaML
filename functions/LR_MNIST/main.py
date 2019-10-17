from dataset_utils import preprocess, partition_training_data
import os
import boto3

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

input_dim = 784
output_dim = 10
model = LogisticRegression(input_dim, output_dim)



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

	# invoke functions 
	payload = dict()
	payload['dataset'] = dataset_name
	payload['data_bucket'] = bucket_name
	payload['model_bucket'] = 'model_bucket'
	payload['num_workers'] = num_workers

	# invoke functions
	for i in range(num_workers):
		payload['rank'] = i
		payload['keys_training_data'] = 'training_{}'.pt.format(i)
		lambda_client = boto3.client('lambda')
		lambda_client.invoke(FunctionName='...', InvocationType='Event', Payload=json.dumps(payload))

