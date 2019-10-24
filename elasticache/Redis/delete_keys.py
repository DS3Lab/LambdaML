# Copyright 2010-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# This file is licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License. A copy of the
# License is located at
#
# http://aws.amazon.com/apache2.0/
#
# This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
# OF ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import logging
import redis
from botocore.exceptions import ClientError


def delete_keys(endpoint, keys):
    """delete the keys in configured redis
    
    :param endpoint: string
    :param keys: list of strings
    :return: True if the reference objects were deleted, otherwise False 
    """
    
    # Connect to redis
    r = redis.Redis(host=endpoint, port=6379, db=0)
    # Set the object
    
    try:
         
         while len(keys)>0:
             r.delete(keys[-1])
             keys.pop()
    except ClinetError as e:
        # AllAccessDisabled error == endpoint not found
        logging.error(e)
        return False
    
    return True

  
def delete_keys_in_hash(endpoint, key, fields):
    """delete the fields within hash key in configured redis
    
    :param endpoint: string
    :param key: string
    :param fields: list of strings
    :return: True if the reference objects were deleted, otherwise False 
    """
    
    # Connect to redis
    r = redis.Redis(host=endpoint, port=6379, db=0)
    # Set the object
    
    try:
         
         while len(fields)>0:
             r.hdel(key, fields[-1])
             fields.pop()
    except ClinetError as e:
        # AllAccessDisabled error == endpoint not found
        logging.error(e)
        return False
    
    return True
        

   

             
            
    
            
        




   

             
            
    
            
        


