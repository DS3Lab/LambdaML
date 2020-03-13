import logging
import memcache
from botocore.exceptions import ClientError

def hdelete_keys(client, fields):
    """delete the field within hash key in configured redis
    
    :param client: memcache client object
    :param key: string
    :param fields: string or list of strings
    :return: True if the reference objects were deleted or don't exsit, otherwise False 
    """
    
    try:
        client.delete_multi(fields)
    except ClientError as e:
        # AllAccessDisabled error ==  client lost
        logging.error(e)
        return False
    
    return True