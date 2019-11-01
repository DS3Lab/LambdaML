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


def list_keys(client):
    """list the keys in configured redis
    
    :param endpoint: string
    :return: List of keys in bytes. If error, return None.
    """
    
    # Connect to redis
    #r = redis.Redis(host=endpoint, port=6379, db=0)
    # Set the object
    try:
        response = client.scan()[1] #match allows for the pattern of keys
        names = [*response[1]]
    except ClientError as e:
        # AllAccessDisabled error == endpoint not found
        logging.error(e)
        return None
    if len(names)>0:
        return names
    return None
        
def hlist_keys(client, key):
    """list all the fields within hash key in configured redis
    
    :param endpoint: string
    :param key: string
    :return: list of keys in bytes, None if error
    """
    
    # Connect to redis
    #r = redis.Redis(host=endpoint, port=6379, db=0)
    # Set the object
    try:
        response = client.hscan(name = key)
        names = [*response[1]]  
    except ClientError as e:
        # AllAccessDisabled error == endpoint or key not found
        logging.error(e)
        return None
    if len(names)>0:
        return names
    return None
        

def handler(event, context):
  
    endpoint = "test.fifamc.ng.0001.euc1.cache.amazonaws.com"
    client = redis.Redis(host=endpoint,port=6379,db=0)
    print(list_keys(client))
    print(hlist_keys(client,"tmp-grads"))
    print(hlist_keys(client,"tmp-updates"))

             
            
    
            
        


