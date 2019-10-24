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


def list_key(endpoint):
    """list the keys in configured redis
    
    :param endpoint: string
    :return: List of keys. If error, return None.
    """
    
    # Connect to redis
    r = redis.Redis(host=endpoint, port=6379, db=0)
    # Set the object
    try:
        response = r.scan()[1] #match allows for the pattern of keys
    except ClinetError as e:
        # AllAccessDisabled error == endpoint not found
        logging.error(e)
        return None
    if len(response)>0:
        return reponse
    return None
        
def list_key_in_hash(endpoint, key):
    """list all the fields within hash key in configured redis
    
    :param endpoint: string
    :param key: string
    :return: True if the reference objects were deleted, otherwise False 
    """
    
    # Connect to redis
    r = redis.Redis(host=endpoint, port=6379, db=0)
    # Set the object
    try:
        r.hkeys(key) 
    except ClinetError as e:
        # AllAccessDisabled error == endpoint or key not found
        logging.error(e)
        return False
    return True
        

   

             
            
    
            
        


