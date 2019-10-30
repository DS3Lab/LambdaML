import time
import urllib.parse
import logging
import boto3

from sync.sync_centroids import *
from s3.get_object import get_object_or_wait
from s3.get_object import get_object
from data_loader.LibsvmDataset import DenseLibsvmDataset, DenseLibsvmDataset2

from model.Kmeans import Kmeans
from sync.sync_meta import SyncMeta

# setting
num_epochs = 100
num_clusters = 10
max_dim = 103
threshold = 0.02
logger = logging.getLogger()
logger.setLevel(logging.INFO)
worker_cent_bucket = "worker-centroids"
avg_cent_bucket = "avg-centroids"


def process_centroid(centroid_byte, nr_cluster, dt, with_error=False):
    cent = np.frombuffer(centroid_byte, dtype=dt)
    logger.info(f"size of cent: {cent.shape}")
    if with_error:
        cent_size = cent.shape[0]-1
        return cent[-1], cent[0:-1].reshape(nr_cluster, int(cent_size / nr_cluster))
    else:
        cent_size = cent.shape[0]
        return cent.reshape(nr_cluster, int(cent_size / nr_cluster))


def lambda_handler(event, context):
    startTs = time.time()
    avg_error = np.iinfo(np.int16).max

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
        dataset_np = DenseLibsvmDataset2(file, max_dim).ins_np
        centroids = dataset_np[0:num_clusters]
        dt = centroids.dtype
        logger.info(f"dtype: {dt}")
        logger.info("Dimension of the dataset: {}, centorids: {}".format(dataset_np.shape, centroids.shape))
        put_object(avg_cent_bucket, "initial", centroids.tobytes())
    else:
        file = get_object(bucket_name, key).read().decode('utf-8').split("\n")
        dataset_np = DenseLibsvmDataset2(file, max_dim).ins_np
        dt = dataset_np.dtype
        cent = get_object_or_wait(avg_cent_bucket, "initial", 5).read()
        if cent is None:
            logger.error("timeout for waiting initial centorids")
            return 0
        centroids = process_centroid(cent, num_clusters, dt)
        logger.info("read data cost {} s".format(time.time() - startTs))

    parse_start = time.time()
    logger.info(f"Dataset size: {dataset_np.shape}")
    logger.info("parse data cost {} s".format(time.time() - parse_start))

    preprocess_start = time.time()
    s3 = boto3.client('s3')

    # Check whether the dimension of the centorids is correct
    if centroids.shape[0] != num_clusters:
        logger.error(f"current centroids has shape {centroids.shape}. Expected {num_clusters}")

    # training
    for epoch in range(num_epochs):
        logger.info(f"{worker_index}-th worker in {epoch}-th epoch")
        if epoch != 0:
            last_epoch = epoch - 1
            cent_with_error = get_object_or_wait(avg_cent_bucket, f"avg-{last_epoch}", 20).read()
            if cent_with_error is None:
                logger.error(f"timeout for waiting centorids for {epoch}")
                return 0
            else:
                logger.info(f"{worker_index}-th worker is reading centroids for {epoch}-th epoch")
                avg_error, centroids = process_centroid(cent_with_error, num_clusters, dt, True)

        if avg_error >= threshold:
            model = Kmeans(dataset_np, centroids)
            model.find_nearest_cluster()
            logger.info(f"Epoch = {epoch} Error = {model.error}")

            sync_start = time.time()
            res = model.centroids.reshape(-1)
            res = np.append(res, model.error)
            success = put_object(worker_cent_bucket, f"{worker_index}_{epoch}", res.tobytes())

            if worker_index == 0 and success:
                compute_average_centroids(avg_cent_bucket, worker_cent_bucket, num_worker, centroids.shape, epoch, dt)

            logger.info("synchronization cost {} s".format(time.time() - sync_start))

        else:
            logger.info(f"{worker_index}-th worker finished training. Error = {avg_error}, centroids = {centroids}")
            return

