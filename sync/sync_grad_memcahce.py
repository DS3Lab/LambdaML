
import urllib

import numpy as np

from elasticache.Memcache.list_keys import hlist_keys
from elasticache.Memcache.get_object import hget_object,hget_object_or_wait
from elasticache.Memcache.set_object import hset_object
from elasticache.Memcache.delete_keys import hdelete_keys


import pickle

def merge_w_b_layers(endpoint, bucket_name, num_workers, prefix):
    #whatever merges w/b grads or model
    num_files = 0 
    merged_value = []
    candidate = []
    split = []
    for worker_index in range(num_workers):
        #split.append(bucket_name+"_"+prefix+str(worker_index))
        #if worker_index%20==0 and worker_index != 0:
        #    candidate.append(split)
        #    split = []
    #candidate.append(split)
        candidate.append(bucket_name+"_"+prefix+str(worker_index))
    #group = len(candidate)
    count = 0
    while num_files < num_workers :#or candidate != []:
        
        objects= hlist_keys(endpoint, candidate)#[count%group])
        
        while objects is not None :#or candidate != []:
            #file_keys = []
            print(objects.keys())
            for key,value in objects.items():
                file_key = key
                #index = file_key.split('_')[2]
                #del candidate[index]
                candidate.remove(file_key)
                data_bytes = value
                data = pickle.loads(data_bytes)
                print("{} is being processed".format(file_key))
                for i in range(len(data)):
                    if num_files == 0:
                        merged_value.append(np.zeros(data[i].shape, dtype=data[i].dtype))
                        
                    merged_value[i] = merged_value[i] + data[i]
                hdelete_keys(endpoint, [file_key])
                
                num_files = num_files + 1
            #file_keys = list(objects.keys())
            
            objects= hlist_keys(endpoint, candidate)#[count%group])
            count = count +1
            print(num_workers-num_files+1)
    # average weights
    if prefix == 'w_':
        merged_value = [value / float(num_workers) for value in merged_value]

    return merged_value

def put_merged_w_b_layers(endpoint, bucket_name, merged_value, prefix, file_postfix):
    # print('put merged weight {} to bucket {}'.format(w_prefix + file_postfix, bucket_name))
    hset_object(endpoint, bucket_name, prefix + file_postfix, pickle.dumps(merged_value))
    # print('put merged bias {} to bucket {}'.format(b_prefix + file_postfix, bucket_name))
    # put_object(bucket_name, b_prefix + file_postfix, b.tobytes())


def get_merged_w_b_layers(endpoint, bucket_name, prefix, file_postfix):
    # print('get merged weight {} in bucket {}'.format(w_prefix + file_postfix, bucket_name))
    merged_value = hget_object_or_wait(endpoint, bucket_name, prefix + file_postfix, 0.001)
    merged_value_np = pickle.loads(merged_value)
    # merged_value_np = np.frombuffer(merged_value, dtype=dtype).reshape(dshape)

    return merged_value_np

def delete_expired_w_b_layers(endpoint,bucket_name, cur_epoch, cur_batch, prefix, end):
    
    expired = [prefix+str(cur_epoch)+""+str(cur_batch-1),prefix+str(cur_epoch-1)+""+str(end-1)]
    
    hdelete_keys(endpoint, bucket_name, expired)











#mergence for single file

   
def merge_w_b_grads(endpoint, bucket_name, num_workers,
                    dtype, w_shape, b_shape,
                    w_grad_prefix="w_grad_", b_grad_prefix="b_grad_"):
    num_w_files = 0
    num_b_files = 0
    w_grad_sum = np.zeros(w_shape, dtype=dtype)
    b_grad_sum = np.zeros(b_shape, dtype=dtype)
    w_candidate = []
    b_candidate = []
    for worker_index in range(num_workers):
        w_candidate.append(bucket_name+"_"+w_grad_prefix+str(worker_index))
        b_candidate.append(bucket_name+"_"+b_grad_prefix+str(worker_index))
    candidate = w_candidate+b_candidate
    while num_w_files < num_workers or num_b_files < num_workers:
        file_keys = []
        objects = hlist_keys(endpoint,candidate)
        
        if objects is not None:
            print(objects.keys())
            for key,value in objects.items():
                #file_key = bytes.decode(obj)
                file_key = key
                file_keys.append(file_key)
                print("the name of the file being processed = {}".format(file_key))
                bytes_data = np.fromstring(value,dtype = dtype)
                if file_key.startswith(bucket_name+"_"+w_grad_prefix):
                    w_grad = bytes_data.reshape(w_shape)
                    #print("merge the {}-th weight grad {} in bucket {} = {}".format(num_w_files, file_key, bucket_name, w_grad[0][:5]))
                    w_grad_sum = w_grad_sum + w_grad
                    num_w_files = num_w_files + 1
                elif file_key.startswith(bucket_name+"_"+b_grad_prefix):
                    b_grad = bytes_data.reshape(b_shape)
                    #print("merge the {}-th bias grad {} in bucket {} = {}".format(num_b_files, file_key, bucket_name, b_grad))
                    b_grad_sum = b_grad_sum + b_grad
                    num_b_files = num_b_files + 1
                #the processed keys are deleted at the stage of mergence
            print("number of file = {},{}".format(num_w_files,num_b_files))
            hdelete_keys(endpoint,file_keys)
            #objects = hlist_keys(endpoint,bucket_name)
            #print("the keys being deleted = {}".format(objects))
        
    return w_grad_sum/num_workers, b_grad_sum/num_workers

                            


def put_merged_w_b_grads(endpoint, bucket_name, w_grad, b_grad,file_postfix,
                        w_grad_prefix="w_grad_", b_grad_prefix="b_grad"):
    print('put merged weight {} to bucket {}'.format(w_grad_prefix+file_postfix, (bucket_name,)))
    hset_object(endpoint, bucket_name,w_grad_prefix+file_postfix, w_grad.tobytes())
    print('put merged bias {} to bucket {}'.format(b_grad_prefix+file_postfix, bucket_name))
    hset_object(endpoint, bucket_name,b_grad_prefix+file_postfix, b_grad.tobytes())


def get_merged_w_b_grads(endpoint, bucket_name, file_postfix,
                        dtype, w_shape, b_shape,
                        w_prefix="w_grad_", b_prefix="b_grad"):
  
    
    #print("get merged weight {} in bucket {}".format(w_prefix+file_postfix , bucket_name))

    w_grad = np.fromstring(hget_object_or_wait(endpoint, bucket_name,  w_prefix + file_postfix , 0.00001), dtype).reshape(w_shape)
    

    #print('get merged bias {} in bucket {}'.format(b_prefix+file_postfix, bucket_name))
    b_grad = np.fromstring(hget_object_or_wait(endpoint, bucket_name, b_prefix + file_postfix, 0.00001), dtype).reshape(b_shape)
    
    return w_grad, b_grad 
def delete_expired_w_b_grads(endpoint, bucket_name, cur_epoch, cur_batch,
                        w_prefix="w_grad_", b_prefix="b_grad"):#,end):
                            
    w_expired = [w_prefix+str(cur_epoch)+""+str(cur_batch-1),w_prefix+str(cur_epoch-1)+""+str(end-1)]
    b_expired = [b_prefix+str(cur_epoch)+""+str(cur_batch-1),b_prefix+str(cur_epoch-1)+""+str(end-1)]
    hdelete_keys(endpoint, bucket_name, w_expired+b_expired)
               

def clear_bucket(endpoint):
    endpoint.flush_all()
    return True 
  
