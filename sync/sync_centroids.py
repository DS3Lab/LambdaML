import urllib
import numpy as np

from s3.list_objects import list_bucket_objects
from s3.get_object import get_object
from s3.put_object import put_object
from s3.delete_objects import delete_objects


def avg_centroids(centroids_vec_list):
    cent_array = np.array(centroids_vec_list)
    return np.average(cent_array, axis=0)


def compute_average_centroids(avg_cent_bucket, worker_cent_bucket, num_workers, shape, epoch):
    num_files = 0
    while num_files < num_workers:
        num_files = 0
        centroids_vec_list = []
        objects = list_bucket_objects(worker_cent_bucket)
        if objects is not None:
            for obj in objects:
                file_key = urllib.parse.unquote_plus(obj["Key"], encoding='utf-8')
                print('file in bucket {} = {}'.format(worker_cent_bucket, file_key))
                data = get_object(worker_cent_bucket, file_key).read()
                tmp_arr = np.frombuffer(data, dtype=dtype).reshape(shape)
                print("the {}-th numpy array".format(num_files))
                print(tmp_arr)
                centroids_vec_list.append(tmp_arr)
                num_files = num_files + 1
        else:
            print('No objects in {}'.format(worker_cent_bucket))
        if num_files < num_workers:
            print("Some workers are not ready.")
        else:
            avg = avg_centroids(centroids_vec_list)
            clear_bucket(worker_cent_bucket)

    print(f"Write averaged centroids {avg} for {epoch}-th epoch to bucket {avg_cent_bucket}")
    put_object(avg_cent_bucket, f"avg-{epoch}", avg.tobytes())
    return 1


def clear_bucket(bucket_name):
    objects = list_bucket_objects(bucket_name)
    if objects is not None:
        file_names = []
        for obj in objects:
            file_key = urllib.parse.unquote_plus(obj["Key"], encoding='utf-8')
            file_names.append(file_key)
        if len(file_names) > 1:
            print("delete files {} in bucket {}".format(file_names, bucket_name))
            delete_objects(bucket_name, file_names)
    return True
