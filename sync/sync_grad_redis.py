
import urllib

import numpy as np

from elasticache.Redis.list_keys import list_keys_in_hash
from elasticache.Redis.get_object import get_object_in_hash, get_object_or_wait_in_hash
from elasticache.Redis.set_object import set_object_in_hash
from elasticache.Redis.delete_key import delete_key_in_hash
from elasticache.Redis.delete_keys import delete_keys_in_hash


def merge_np_bytes(endpoint, bucket_name, num_workers, dtype, shape):
    num_files = 0
    sum_arr = np.zeros(shape, dtype=dtype)

    while num_files < num_workers:
        #objects = list_bucket_objects(bucket_name)
        objects = list_keys_in_hash(endpoint,bucket_name)
        if objects is not None:
            for obj in objects:
                file_key = urllib.parse.unquote_plus(obj["Key"], encoding='utf-8')
                print('file in bucket {} = {}'.format(bucket_name, file_key))
                #data = get_object(bucket_name, file_key).read()
                #tmp_arr = np.frombuffer(data, dtype=dtype).reshape(shape)
                tmp_arr = np.frombuffer(get_object_in_hash(endpoint, bucket_name, file_key), dtype).reshape(shape)
                print("the {}-th numpy array".format(num_files))
                print(tmp_arr)
                sum_arr = sum_arr + tmp_arr
                num_files = num_files + 1
                #delete_object(bucket_name, file_key)
                delete_key_in_hash(endpoint,bucket_name,file_key)
        else:
            # Didn't get any keys
            print('No objects in {}'.format(bucket_name))

    return sum_arr


def merge_w_b_grads(endpoint, num_workers,
                    dtype, w_shape, b_shape,
                    w_grad_prefix="w_grad_", b_grad_prefix="b_grad"):
    num_w_files = 0
    num_b_files = 0
    w_grad_sum = np.zeros(w_shape, dtype=dtype)
    b_grad_sum = np.zeros(b_shape, dtype=dtype)

    while num_w_files < num_workers or num_b_files < num_workers:
        #objects = list_bucket_objects(bucket_name)
        objects = list_keys_in_hash(endpoint,bucket_name)
        if objects is not None:
            for obj in objects:
                file_key = urllib.parse.unquote_plus(obj["Key"], encoding='utf-8')
                #data = get_object(bucket_name, file_key).read()
                #bytes_data = np.frombuffer(data, dtype=dtype)
                bytes_data = np.fromstring(get_object(endpoint, bucket_name, file_key), dtype)
                if file_key.startswith(w_grad_prefix):
                    w_grad = bytes_data.reshape(w_shape)
                    print("merge the {}-th weight grad {} in bucket {} = {}".format(num_w_files, file_key, bucket_name, w_grad[0][:5]))
                    w_grad_sum = w_grad_sum + w_grad
                    num_w_files = num_w_files + 1
                elif file_key.startswith(b_grad_prefix):
                    b_grad = bytes_data.reshape(b_shape)
                    print("merge the {}-th bias grad {} in bucket {} = {}".format(num_b_files, file_key, bucket_name, b_grad))
                    b_grad_sum = b_grad_sum + b_grad
                    num_b_files = num_b_files + 1
                #delete_object(bucket_name, file_key)
                delete_object_in_hash(endpoint,bucket_name,file_key)
        # else:
        #     # Didn't get any keys
        #     print('No objects in {}'.format(bucket_name))

    return w_grad_sum, b_grad_sum


def put_merged_w_b_grad(endpoint, bucket_name, w_grad, b_grad, file_postfix,
                        w_grad_prefix="w_grad_", b_grad_prefix="b_grad"):
    print('put merged weight {} to bucket {}'.format(w_grad_prefix + file_postfix, bucket_name))
    #put_object(bucket_name, w_grad_prefix + file_postfix, w_grad.tobytes())
    set_object_in_hash(endpoint, bucket_name,w_grad_prefix + file_postfix, w_grad.tobytes())
    print('put merged bias {} to bucket {}'.format(b_grad_prefix + file_postfix, bucket_name))
    #put_object(bucket_name, b_grad_prefix + file_postfix, b_grad.tobytes())
    set_object_in_hash(endpoint, bucket_name,b_grad_prefix + file_postfix, b_grad.tobytes())


def get_merged_w_b_grad(endpoint, bucket_name, file_postfix,
                        dtype, w_shape, b_shape,
                        w_prefix="w_grad_", b_prefix="b_grad", dtype = 'float'):
   
    
    print('get merged weight {} in bucket {}'.format(w_prefix + file_postfix, bucket_name))
    #w_data = get_object_or_wait(bucket_name, w_prefix + file_postfix, 0.1).read()
    w_grad = np.fromstring(get_object_or_wait_in_hash(endpoint, bucket_name, w_prefix + file_postfix, 0.1), dtype).reshape(w_shape)
    #w_grad = np.frombuffer(w_data, dtype=dtype).reshape(w_shape)

    print('get merged bias {} in bucket {}'.format(b_prefix + file_postfix, bucket_name))
    #b_data = get_object_or_wait(bucket_name, b_prefix + file_postfix, 0.1).read()
    #b_grad = np.frombuffer(b_data, dtype=dtype).reshape(b_shape)
    b_grad = np.fromstring(get_object_or_wait_in_hash(endpoint, bucket_name, b_prefix + file_postfix, 0.1), dtype).reshape(w_shape)

    return w_grad, b_grad


def delete_expired_w_b(endpoint, bucket_name, cur_epoch, cur_batch,
                        w_prefix="w_grad_", b_prefix="b_grad"):
    objects = list_bucket_objects(bucket_name)
    if objects is not None:
        for obj in objects:
            file_key = urllib.parse.unquote_plus(obj["Key"], encoding='utf-8')
            if file_key.startswith(w_prefix) or file_key.startswith(b_prefix):
                key_splits = file_key.split("_")
                key_batch = int(key_splits[-1])
                key_epoch = int(key_splits[-2])
                if key_epoch < cur_epoch or (key_epoch == cur_epoch and key_batch < cur_batch):
                    print("delete object {} in bucket {}".format(file_key, bucket_name))
                    #delete_object(bucket_name, file_key)
                    delete_key_in_hash(endpoint, bucket_name, file_key)
                # elif key_epoch == cur_epoch and key_batch < cur_batch:
                #     delete_object(bucket_name, file_key)


def clear_bucket(endpoint, bucket_name):
    #objects = list_bucket_objects(bucket_name)
    objects = list_keys_in_hash(endpoint, bucket_name)
    if objects is not None:
        file_names = []
        for obj in objects:
            #file_key = urllib.parse.unquote_plus(obj["Key"], encoding='utf-8')
            file_key = obj
            file_names.append(file_key)
        if len(file_names) > 1:
            print("delete files {} in bucket {}".format(file_names, bucket_name))
            #delete_objects(bucket_name, file_names)
            delet_keys_in_hash(endpoint, bucket_name, file_names)
    return True

