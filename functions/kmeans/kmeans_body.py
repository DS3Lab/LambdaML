import math
import numpy as np
import boto3
import logging


logger = logging.getLogger()
logger.setLevel(logging.INFO)

def l2_dist(a, b, ax=1):
	return np.linalg.norm(a - b, axis=ax)


def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)


def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])


def find_nearest_cluster(X, centroids_vec):
	closest = closest_centroid(X, centroids_vec)
	new_centroids_vec = move_centroids(X, closest, centroids_vec)
	error = l2_dist(centroids_vec, new_centroids_vec, None)
	return (error, X, new_centroids_vec)


def avg_centroids(centroids_vec_list):
	cent_array = np.array(centroids_vec_list)
	return np.average(cent_array, axis=0)


def update_intermediate_result(s3, intermediate_result, bucket_name, file_name):
	tmp = ""
	for i in intermediate_result:
		tmp = tmp + "{}\n".format(i)
	s3.put_object(Body=str.encode(tmp), Bucket=bucket_name, Key=file_name)


def sync_result_multiple_files(s3, bucket_subfolder_name, current_iter, current_cent, nth):
	"""
	Synchronize results from [nr_workers] files under bucket_subfolder_name.
	The nth file stores:
		epoch
		centroid_from_worker_n
	Iterate through each file and check whether
	"""
	nth_worker_result = f"{current_iter}\n{current_cent}"
	centroids_vec_list = []
	s3.put_object(Body=str.encode(nth_worker_result), Bucket=bucket_subfolder_name, Key=nth)
	response = s3.list_objects_v2(Bucket=bucket_subfolder_name)
	for object in response['Contents']:
		# still not safe since other workers might try to modify the file at the same time
		if object['Key'] == "avg":
			continue
		file = s3.get_object(Bucket=bucket_name, Key=object['Key'])
		body = file['Body'].read().decode().split("\n")
		epoch, centroid = body[0], body[1]
		if (epoch < current_iter):
			# other workers have not yet updated the result for this epoch
			return (False, current_cent)
		else:
			centroids_vec_list.append(centroid)
	# all workers have updated the result for this epoch
	avg_centroids = avg_centroids(centroids_vec_list)
	avg_result = f"{current_iter}\n{avg_centroids}"
	s3.put_object(Body=str.encode(avg_result), Bucket=bucket_subfolder_name, Key="avg")
	return (True, avg_centroids)


def sync_result_single_file(s3, bucket_name, file_name, nr_workers, current_iter, current_cent, nth):
	"""
	Synchronize intermediate result from a single file which stores:
		count
		epoch
		centroid_from_worker_1
		centroid_from_worker_2
		...
		centroid_from_worker_n
		average_centroid_of_last_iteration
	"""
	s3_object = s3.get_object(Bucket=bucket_name, Key=file_name)
	body = s3_object['Body'].read().decode()
	intermediate_result = body.split("\n")
	count, iteration = intermediate_result[0], intermediate_result[1]
	if (iteration == current_iter):
		intermediate_result[0] += 1
		intermediate_result[nth-2] = current_cent
		update_intermediate_result(s3, intermediate_result, bucket_name, file_name)
		return True, intermediate_result[nr_workers+1]
	elif (iteration < current_iter): # current process has already finished [iteration]-th sync
		if (count == nr_workers) : # others are ready
			# update the iteration and clear the count
			intermediate_result[0] = 0
			intermediate_result[1] += 1
			intermediate_result[nr_workers+1] = avg_centroids(intermediate_result[2:2+nr_workers])
			data = "{}\n{}\n{}\n".format(count, iteration, new_centroids)
			s3.put_object(Body=str.encode(data), Bucket=bucket_name, Key=file_name)
			return True, intermediate_result[nr_workers+1]
		else:
			# need to wait for others to finish the last round
			return False, intermediate_result[nr_workers+1]
	else:
		# iteration > current_iter
		# this should not happen
		logger.info("Something wrong.")
		return True, intermediate_result[nr_workers+1]



def lambda_handler(event, context):
	tolerance = event['tolerance']
	X = event['X'].split("\n") # X: training data. Need to agree on the format in the future
	c = [i.split(",") for i in event['centroids_vec']]
	centroids_vec = np.asarray(c, dtype=float)
	error = 10000
	s3 = boto3.client('s3')
	iter = 1
	batch = event["training_batch_size"]
	file_name = "kmeans_intermediate_result.txt"
	bucket_subfolder_name = f"{event['bucket_name']}/kmeans_intermediate"

	data = [s.split(",") for s in X[1:] if s != '']
	np_data = np.array(data, dtype=float)

	cluster_label_vec = [0 for i in range(len(X))]

	row, column = np_data.shape
	logger.info(f"{row} rows and {column} columns.")

	while (error > tolerance):
		if (iter % batch == 0):
			current_iter = int(iter/batch) # starts from 1
			success = False
			while (not success):
				success, centroids_vec = sync_result_multiple_files(s3, bucket_subfolder_name,
																current_iter, centroids_vec, event["nth_worker"])

		error, X, centroids_vec = find_nearest_cluster(np_data, centroids_vec)
		iter += 1

	logger.info("Error :{}".format(error))
	return error
