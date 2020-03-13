import urllib.parse
#import boto3
import logging
import time
import numpy as np

from sync.sync_centroids_memcached import compute_average_centroids
from elasticache.Memcached.get_object import hget_object_or_wait
from elasticache.Memcached.set_object import hset_object
from s3.get_object import get_object
from elasticache.Memcached.__init__ import memcached_init
from data_loader.LibsvmDataset import DenseLibsvmDataset, SparseLibsvmDataset
from sync.sync_meta import SyncMeta
from functions.kmeans.utils import store_centroid_as_numpy, process_centroid, get_new_centroids

# setting
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    avg_error = np.iinfo(np.int16).max

    num_features = event['num_features']
    num_clusters = event['num_clusters']
    worker_cent_bucket = event["worker_cent_bucket"]
    avg_cent_bucket = event["avg_cent_bucket"]
    num_epochs = event["num_epochs"]
    threshold = event["threshold"]
    dataset_type = event["dataset_type"]
    elastic_location = event["elasticache"]
    elastic_endpoint = memcached_init(elastic_location)
    Reading data from S3
    bucket_name = event['bucket_name']
    key = urllib.parse.unquote_plus(event['key'], encoding='utf-8')
    logger.info(f"Reading training data from bucket = {bucket_name}, key = {key}")
    key_splits = key.split("_")
    worker_index = 0
    num_worker = 1
    sync_meta = SyncMeta(worker_index, num_worker)
    logger.info(f"Synchronizing meta {sync_meta.__str__()}")

    event_start = time.time()
    file = get_object(bucket_name, key).read().decode('utf-8').split("\n")
    s3_end = time.time()
    logger.info(f"Getting object from s3 takes {s3_end - event_start}s")

    if dataset_type == "dense":
        # dataset is stored as numpy array
        dataset = DenseLibsvmDataset2(file, num_features).ins_np
        print(len(dataset))
        dt = dataset.dtype
        centroid_shape = (num_clusters, dataset.shape[1])
    else:
        # dataset is sparse, stored as sparse tensor
        dataset = SparseLibsvmDataset(file, num_features)
        first_entry = dataset.ins_list[0].to_dense().numpy()
        dt = first_entry.dtype
        centroid_shape = (num_clusters, first_entry.shape[1])
    parse_end = time.time()
    #logger.info(f"Parsing dataset takes {parse_end - s3_end}s")
    logger.info(
        f"Dataset: {dataset_type}, dtype: {dt}. Centroids shape: {centroid_shape}. num_features: {num_features}")

    if worker_index == 0:
        if dataset_type == "dense":
            centroids = dataset[0:num_clusters].reshape(-1)
        else:
            centroids = store_centroid_as_numpy(dataset.ins_list[0:num_clusters], num_clusters)
        hset_object(elastic_endpoint, avg_cent_bucket, "initial", centroids.tobytes())
    else:
        cent = hget_object_or_wait(elastic_endpoint, avg_cent_bucket, "initial", 0.1)
        centroids = process_centroid(cent, num_clusters, dt)
        if centroid_shape != centroids.shape:
            logger.error("The shape of centroids does not match.")
        logger.info(f"Waiting for initial centroids takes {time.time() - parse_end} s")

    s3 = boto3.client('s3')
    training_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        logger.info(f"{worker_index}-th worker in {epoch}-th epoch")
        if epoch != 0:
            last_epoch = epoch - 1
            cent_with_error = hget_object_or_wait(elastic_endpoint, avg_cent_bucket, f"avg-{last_epoch}", 0.1)
            wait_end = time.time()
            if worker_index != 0:
                logger.info(f"Wait for centroid for {epoch}-th epoch. Takes {wait_end - epoch_start}")
            avg_error, centroids = process_centroid(cent_with_error, num_clusters, dt, True)

        if avg_error >= threshold:
            res = get_new_centroids(dataset, dataset_type, centroids, epoch, num_features, num_clusters, dt)
            dt = res.dtype
            success = hset_object(elastic_endpoint, worker_cent_bucket, f"{worker_index}_{epoch}", res.tobytes())

            if worker_index == 0 and success:
                sync_start = time.time()
                compute_average_centroids(elastic_endpoint,avg_cent_bucket, worker_cent_bucket, num_worker, centroids.shape, epoch, dt)
                logger.info(f"Waiting for all workers takes {time.time() - sync_start} s")

        else:
            logger.info(f"{worker_index}-th worker finished training. Error = {avg_error}, centroids = {centroids}")
            logger.info(f"Whole process time : {time.time() - training_start}")
            return

if __name__=="__main__":
    import sys
    sys.path.append('/Users/liuyue/LambdaML')

    import json
    payload = dict()
    payload['num_features'] = 128
    payload['num_clusters'] = 1
    payload['worker_cent_bucket'] = "tmp-updates"
    payload['avg_cent_bucket'] = "model"
    payload['num_epochs'] = 1
    payload["threshold"] = 1
    payload["dataset_type"] = "dense"
    payload["elasticache"] = "test.fifamc.cfg.euc1.cache.amazonaws.com"
    lambda_handler(payload,None)
