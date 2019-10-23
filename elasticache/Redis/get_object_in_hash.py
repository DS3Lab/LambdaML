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
import time
import redis
from botocore.exceptions import ClientError


def get_object_in_hash(endpoint, key,field, dtype):
    """Retrieve an object from configured redis under specified key
    
    :param endpoint: string
    :param key: string
    :param field: string
    :param dtype: string
    :return: string. If error, return None.
    """
    # Connect to redis
    r = redis.Redis(host=endpoint, port=6379, db=0)
    # Retrieve the object
    try:
        response = r.hget(key=key,field=field)
    except ClinetError as e:
        # AllAccessDisabled error == endpoint or key not found
        logging.error(e)
        return None
    return response

def get_object_or_wait(endpoint, key, sleep_time):
    """Retrieve an object from configured redis under specified key
    
    :param endpoint: string
    :param key: string
    :param dtype: string
    :return: numpy arrary. If error, return None.
    """
    # Connect to redis
    r = redis.Redis(host=endpoint, port=6379, db=0)
    # Retrieve the object
    while True:
        try:
            response = r.hget(key=key,field=field)
            return response
        except ClinetError as e:
            # AllAccessDisabled error == endpoint or key not found
            time.sleep(sleep_time)
   

             
            
    
            
        

