import time
import urllib.parse
import logging
import boto3

from sync.sync_centroids_elastic import *
from elasticache.Redis.get_object import hget_object_or_wait
from elasticache.Redis.set_object import hset_object
from s3.get_object import get_object
from elasticache.Redis.__init__ import redis_init
from data_loader.LibsvmDataset import DenseLibsvmDataset, DenseLibsvmDataset2

from model.Kmeans import Kmeans
from sync.sync_meta import SyncMeta

# setting
logger = logging.getLogger()
logger.setLevel(logging.INFO)
redis_location = "test-001.fifamc.0001.euc1.cache.amazonaws.com"
elastic_endpoint = redis_init(redis_location)


def process_centroid(centroid, nr_cluster, dt, with_error=False):
    cent = np.frombuffer(centroid, dtype=dt)
    logger.info(f"size of cent: {cent.shape}")
    if with_error:
        cent_size = cent.shape[0] - 1
        return cent[-1], cent[0:-1].reshape(nr_cluster, int(cent_size / nr_cluster))
    else:
        cent_size = cent.shape[0]
        return cent.reshape(nr_cluster, int(cent_size / nr_cluster))


def lambda_handler(event, context):
    startTs = time.time()
    avg_error = np.iinfo(np.int16).max

    max_dim = event['max_dim']
    num_clusters = event['num_clusters']
    worker_cent_bucket = event["worker_cent_bucket"]
    avg_cent_bucket = event["avg_cent_bucket"]
    num_epochs = event["num_epochs"]
    threshold = event["threshold"]

    # Reading data from S3
    bucket_name = event['bucket_name']
    key = urllib.parse.unquote_plus(event['key'], encoding='utf-8')

    logger.info(f'Reading training data from bucket = {bucket_name}, key = {key}')

    key_splits = key.split("_")
    worker_index = int(key_splits[0])

    num_worker = int(key_splits[1])
    sync_meta = SyncMeta(worker_index, num_worker)
    logger.info("synchronization meta {}".format(sync_meta.__str__()))

    # read file from s3
    if worker_index == 0:
        file = get_object(bucket_name, key).read().decode('utf-8').split("\n")
        read_end = time.time()
        logger.info("read data cost {} s".format(read_end - startTs))
        dataset_np = DenseLibsvmDataset2(file, max_dim).ins_np
        centroids = dataset_np[0:num_clusters]
        dt = centroids.dtype
        logger.info(f"dtype: {dt}")
        logger.info("parse data cost {} s".format(time.time() - read_end))
        logger.info("Dimension of the dataset: {}, centorids: {}".format(dataset_np.shape, centroids.shape))
        logger.info(f"Putting initial centorids: {centroids.shape}")
        hset_object(elastic_endpoint, avg_cent_bucket, "initial", centroids.tobytes())
        cent = hget_object_or_wait(elastic_endpoint, avg_cent_bucket, "initial", 0.1)
        centroids = process_centroid(cent, num_clusters, dt)
        logger.info(f"Getting Initial centorids: {centroids.shape}")

    else:
        file = get_object(bucket_name, key).read().decode('utf-8').split("\n")
        read_end = time.time()
        logger.info("read data cost {} s".format(read_end - startTs))
        dataset_np = DenseLibsvmDataset2(file, max_dim).ins_np
        dt = dataset_np.dtype
        wait_cent_start = time.time()
        cent = hget_object_or_wait(elastic_endpoint, avg_cent_bucket, "initial", 0.1)
        if cent is None:
            logger.error("timeout for waiting initial centorids")
            return 0
        centroids = process_centroid(cent, num_clusters, dt)
        logger.info(f"Initial centorids: {centroids.shape}")
        logger.info("Waiting for initial centroids cost {} s".format(time.time() - wait_cent_start))

    preprocess_start = time.time()
    s3 = boto3.client('s3')

    # training
    for epoch in range(num_epochs):
        epoch_start = time.time()
        logger.info(f"{worker_index}-th worker in {epoch}-th epoch")
        if epoch != 0:
            last_epoch = epoch - 1
            cent_with_error = hget_object_or_wait(elastic_endpoint, avg_cent_bucket, f"avg-{last_epoch}", 0.1)
            wait_time = time.time()
            logger.info(f"Wait for centroid for {epoch}-th epoch. Takes {wait_time - epoch_start}")
            if cent_with_error is None:
                logger.error(f"timeout for waiting centorids for {epoch}")
                return 0
            else:
                logger.info(f"{worker_index}-th worker is reading centroids for {epoch}-th epoch")
                avg_error, centroids = process_centroid(cent_with_error, num_clusters, dt, True)

        if avg_error >= threshold:
            compute_start = time.time()
            model = Kmeans(dataset_np, centroids)
            model.find_nearest_cluster()
            compute_end = time.time()
            logger.info(f"Epoch = {epoch} Error = {model.error}. Computation time: {compute_end - compute_start}")

            res = model.centroids.reshape(-1)
            res = np.append(res, model.error)
            success = hset_object(elastic_endpoint, worker_cent_bucket, f"{worker_index}_{epoch}", res.tobytes())

            if not success:
                logger.info(
                    f"{worker_index}-th worker does not successfully put the result to {worker_cent_bucket} for {epoch}-epoch")
                return

            if worker_index == 0 and success:
                sync_start = time.time()
                compute_average_centroids(elastic_endpoint, avg_cent_bucket, worker_cent_bucket, num_worker,
                                          centroids.shape, epoch, dt)
                logger.info("Wait for all workers cost {} s".format(time.time() - sync_start))

        else:
            logger.info(f"{worker_index}-th worker finished training. Error = {avg_error}, centroids = {centroids}")
            logger.info(f"Whole process time : {time.time() - preprocess_start}")
            return

