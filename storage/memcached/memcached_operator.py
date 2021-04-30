import logging
import time
import memcache
from botocore.exceptions import ClientError


def _get_client(ip, port=11211):
    try:
        endpoint = ip + ":" + str(port)
        client = memcache.Client([endpoint])
    except ClientError as e:
        # AllAccessDisabled error == endpoint not found
        logging.error(e)
        return False
    return client


def _delete_key(client, key):
    """Delete key in configured memcached

    :param client: memcached client object
    :param key: string
    :return: True if the reference objects were deleted or don't exist, otherwise False
    """

    try:
        client.delete(key)
    except ClientError as e:
        # AllAccessDisabled error ==  client lost
        logging.error(e)
        return False
    return True


def _delete_keys(client, keys):
    """Delete keys in configured memcached

    :param client: memcached client object
    :param keys: list of strings
    :return: True if the reference objects were deleted or don't exist, otherwise False
    """

    try:
        client.delete_multi(keys)
    except ClientError as e:
        # AllAccessDisabled error ==  client lost
        logging.error(e)
        return False
    return True


def _clear_all(client):
    """Delete all keys in configured memcached

    :param client: memcached client object
    """
    client.flust_all()
    return True


def _get_object(client, key):
    """Retrieve an object from configured memcached under specified key

    :param client: memcached client object
    :param key: string
    :return: direct bytestream, no need to read from buffer. If error, return None.
    """
    try:
        response = client.get(key=key)
    except ClientError as e:
        # AllAccessDisabled error == client lost
        logging.error(e)
        return None
    return response


def _get_object_v2(client, key, field):
    """Retrieve an object from configured memcached under specified key and field

    :param client: memcached client object
    :param key: string
    :param field: string
    :return: direct bytestream, no need to read from buffer. If error, return None.
    """
    try:
        mem_key = key + "_" + field
        response = client.get(key=mem_key)
    except ClientError as e:
        # AllAccessDisabled error == client lost
        logging.error(e)
        return None
    return response


def _get_objects(client, keys):
    """Retrieve objects from configured memcached under specified keys

    :param client: memcache client object
    :param keys: list of strings
    :return: direct bytestream, no need to read from buffer. If error or timeout, return None.
    """
    objects = client.get_multi(keys)
    if bool(objects):
        return objects
    else:
        return None


def _get_object_or_wait(client, key, sleep_time, time_out=60):
    """Retrieve an object from configured memcache under specified key; wait if not exist

    :param client: memcache client object
    :param key: string
    :param sleep_time: float
    :param time_out: int (seconds)
    :return: direct bytestream, no need to read from buffer. If error or timeout, return None.
    """
    try:
        start_time = time.time()
        while True:
            if time.time() - start_time > time_out:
                return None
            response = client.get(key=key)
            if response is not None:
                return response
            time.sleep(sleep_time)
    except ClientError as e:
        # AllAccessDisabled error == client lost
        logging.error(e)
        return None


def _get_object_or_wait_v2(client, key, field, sleep_time, time_out=60):
    """Retrieve an object from configured memcache under specified key and field; wait if not exist

    :param client: memcache client object
    :param key: string
    :param field: string
    :param sleep_time: float
    :param time_out: int (seconds)
    :return: direct bytestream, no need to read from buffer. If error or timeout, return None.
    """
    try:
        start_time = time.time()
        mem_key = key + "_" + field
        while True:
            if time.time() - start_time > time_out:
                return None
            response = client.get(key=mem_key)
            if response is not None:
                return response
            time.sleep(sleep_time)
    except ClientError as e:
        # AllAccessDisabled error == client lost
        logging.error(e)
        return None


def _set_object(client, key, src_data):
    """Add value from configured memcache under specified key of certain hashtable

    :param client: string
    :param key: string
    :param src_data: bytestream, int, string
    :return: True if src_data was added to memcache at client under key, otherwise False
    """
    if isinstance(src_data, bytes) or isinstance(src_data, str) or isinstance(src_data, int):
        object_data = src_data
    else:
        logging.error('Type of ' + str(type(src_data)) +
                      ' for the argument \'src_data\' is not supported.')

    try:
        response = client.set(key, src_data)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def _set_object_v2(client, key, field, src_data):
    """Add value from configured memcache under specified key of certain hashtable

    :param client: string
    :param field: string
    :param key: string
    :param src_data: bytestream, int, string
    :return: True if src_data was added to memcache at client under key, otherwise False
    """
    if isinstance(src_data, bytes) or isinstance(src_data, str) or isinstance(src_data, int):
        object_data = src_data
    else:
        logging.error('Type of ' + str(type(src_data)) +
                      ' for the argument \'src_data\' is not supported.')

    try:
        mem_key = key + "_" + field
        response = client.set(mem_key, src_data)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def _list_keys(client, keys):
    """

    :param client: string
    :param keys: list of candidate keys
    :return: True if all keys exist, None otherwise
    """
    objects = client.client.get_multi(keys)
    if bool(objects):
        return objects
    else:
        return None