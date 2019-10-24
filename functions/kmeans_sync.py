import time
import urllib.parse
import logging
import boto3

from sync.sync_centroids import *
from s3.get_object import get_object_or_wait
from s3.get_object import get_object
from data_loader.LibsvmDataset import DenseLibsvmDataset

from model.Kmeans import Kmeans
from sync.sync_meta import SyncMeta

# setting
random_seed = 42
num_epochs = 200
num_clusters = 10
logger = logging.getLogger()
logger.setLevel(logging.INFO)
worker_cent_bucket = "worker_centroids"
avg_cent_bucket = "avg_centroids"


def process_centroid(centroid_byte, n):
    cent = np.frombuffer(centroid_byte)
    cent_size = cent.shape[0]
    return cent.reshape(int(cent_size/n), n)


def lambda_handler(event, context):
    startTs = time.time()

    # Reading data from S3
    bucket_name = event['bucket_name']
    key = urllib.parse.unquote_plus(event['key'], encoding='utf-8')

    logger.info(f'Reading training data from bucket = {bucket_name}, key = {key}')

    key_splits = key.split("_")
    worker_index = int(key_splits[0])

    num_worker = int(key_splits[1])
    sync_meta = SyncMeta(worker_index, num_worker)
    logger.info("synchronization meta {}".format(sync_meta.__str__()))
    centroids = np.array()

    # read file from s3
    if worker_index == 0:
        file = get_object(bucket_name, key).read().decode('utf-8').split("\n")
        dataset_np = DenseLibsvmDataset(file).ins_list_np
        centroids = dataset_np[0:num_clusters]
        put_object(avg_cent_bucket, "initial", centroids.tobytes())
    else:
        cent = get_object_or_wait(avg_cent_bucket, "initial", 100).read()
        if cent is None:
            logger.error("timeout for waiting initial centorids")
            return 0
        centroids = process_centroid(cent)
        file = get_object(bucket_name, key).read().decode('utf-8').split("\n")
        dataset_np = DenseLibsvmDataset(file).ins_list_np
        logger.info("read data cost {} s".format(time.time() - startTs))

    parse_start = time.time()
    logger.info(f"Dataset size: {dataset_np.shape}")
    logger.info("parse data cost {} s".format(time.time() - parse_start))

    preprocess_start = time.time()
    s3 = boto3.client('s3')

    # Check whether the dimension of the centorids is correct
    if centroids.shape[1] != num_clusters:
        logger.error(f"current centroids has shape {centroids.shape}. Expected {num_clusters}")

    # training
    for epoch in range(num_epochs):
        if epoch != 0:
            last_epoch = epoch - 1
            cent = get_object_or_wait(avg_cent_bucket, f"avg_{last_epoch}", 100)
            if cent is None:
                logger.error(f"timeout for waiting centorids for {epoch}")
                return 0
            centroids = process_centroid(cent)

        model = Kmeans(dataset_np, centroids)
        logger.info(f"Epoch = {epoch} Error = {model.error}")

        sync_start = time.time()
        put_object(worker_cent_bucket, f"{worker_index}_{epoch}", model.centroids.tobytes())

        if worker_index == 0:
            compute_average_centroids(avg_cent_bucket, worker_cent_bucket, num_worker, centroids.shape, epoch);

        logger.info("synchronization cost {} s".format(time.time() - sync_start))
