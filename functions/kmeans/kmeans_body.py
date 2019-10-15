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


def s3_check(s3, bucket_name, file_name, nr_workers, current_iter, current_cent, nth):
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
	nth = event["nth_worker"]
	nr_workers = event["nr_workers"]
	tolerance = event['tolerance']
	X = event['X'].split("\n")
	c = [i.split(",") for i in event['centroids_vec']]
	centroids_vec = np.asarray(c, dtype=float)
	error = 10000
	s3 = boto3.client('s3')
	iter = 1
	batch = event["training_batch_size"]
	file_name = "kmeans_intermediate_result.txt"

	data = [s.split(",") for s in X[1:] if s != '']
	np_data = np.array(data, dtype=float)

	cluster_label_vec = [0 for i in range(len(X))]

	row, column = np_data.shape
	logger.info("{} rows and {} columns.".format(row, column))

	while (error > tolerance):
		if (iter % batch == 0):
			current_iter = int(iter/batch) # starts from 1
			success = False
			while (not success):
				success, centroids_vec = s3_check(s3, event['bucket_name'], file_name,
												nr_workers, current_iter, centroids_vec, nth)

		error, X, centroids_vec = find_nearest_cluster(np_data, centroids_vec)
		iter += 1

	logger.info("Error :{}".format(error))
	return error
