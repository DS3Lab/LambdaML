import boto3
import json
import logging
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
	payload = dict()
	nr_of_training_files = event["nr_of_training_files"]
	training_file_name = event["training_file_name"]
	training_file_extension = event["training_file_extension"]
	training_file_list = ["{}{}.{}".format(training_file_name, i, training_file_extension) for i in range(nr_of_training_files)]

	s3 = boto3.client('s3')
	training_data = []
	for file in training_file_list:
		logger.info("Loading {}".format(file))
		s3_object = s3.get_object(Bucket=event["bucket_name"], Key=file)
		body = s3_object['Body'].read().decode()
		training_data.append(body)

	payload["tolerance"] = event["tolerance"]
	payload["k_clusters"] = event["k_clusters"]
	payload["bucket_name"] = event["bucket_name"]
	payload["training_batch_size"] = event["training_batch_size"]
	payload["nr_workers"] = nr_of_training_files

	#initialize centroids_vec
	first = training_data[0].split("\n")
	payload['centroids_vec'] = first[1:payload["k_clusters"]+1]
	payload['cluster_label_vec'] = [j for j in range(nr_of_training_files)]

	#initialize the file for storing intermediate result

	intermediate_result = "0\n1\n" # count = 0, iteration = 1
	for i in range(nr_of_training_files):
		intermediate_result += "\n"
	intermediate_result += "{}\n".format(payload['centroids_vec'])
	intermediate_result = str.encode(intermediate_result)
	s3.put_object(Body=intermediate_result, Bucket=event["bucket_name"], Key="kmeans_intermediate_result.txt")

	for i in range(nr_of_training_files):
		logger.info("Invoke kmeans for each training file...")
		payload['X'] = training_data[i]
		payload['nth_worker'] = i
		payload["final_file_name"] = "kmeans_final_{}.txt".format(i)

		lambda_client = boto3.client('lambda')
		lambda_client.invoke(FunctionName='kmeans_body',
									  InvocationType='Event',
									  Payload=json.dumps(payload))



	return
