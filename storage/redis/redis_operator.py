import logging
import time
import redis
from botocore.exceptions import ClientError


def _get_client(endpoint, port):
    client = redis.Redis(host=endpoint, port=port, db=0)
    return client


def _delete_keys(client, keys):
    """Delete keys in configured redis

    :param client: redis client object
    :param keys: string or list of strings
    :return: True if the reference objects were deleted or don't exist, otherwise False
    """

    try:
        client.delete(*keys)
    except ClientError as e:
        # AllAccessDisabled error == client lost
        logging.error(e)
        return False
    return True


def _delete_fields(client, key, fields):
    """delete the field within hash key in configured redis

    :param client: redis client object
    :param key: string
    :param fields: string or list of strings
    :return: True if the reference objects were deleted or don't exist, otherwise False
    """

    try:
        client.hdel(key, *fields)
    except ClientError as e:
        # AllAccessDisabled error ==  client lost
        logging.error(e)
        return False

    return True


def _get_object(client, key):
    """Retrieve an object from configured redis under specified key

    :param client: redis client object
    :param key: string
    :return: direct bytestream, no need to read from buffer. If error, return None.
    """

    # Retrieve the object
    try:
        response = client.get(name=key)
    except ClientError as e:
        # AllAccessDisabled error == client lost
        logging.error(e)
        return None
    return response


def _get_object_or_wait(client, key, sleep_time, time_out=60):
    """Retrieve an object from configured redis under specified key; wait if not exist

    :param client: Redis client object
    :param key: string
    :param sleep_time: float
    :param time_out: int (seconds)
    :return: direct bytestream, no need to read from buffer. If error or timeout, return None.
    """
    # Connect to redis
    # client = redis.Redis(host=endpoint, port=6379, db=0)

    # Retrieve the object
    try:
        start_time = time.time()
        while True:
            if time.time() - start_time > time_out:
                return None
            response = client.get(name=key)
            if response is not None:
                return response
            time.sleep(sleep_time)
    except ClientError as e:
        # AllAccessDisabled error == client lost
        logging.error(e)
        return None


def _hget_object(client, key, field):
    """Retrieve an object from configured redis under specified key and field

    :param client: redis client object
    :param key: string
    :param field: string
    :return: direct bytestream, no need to read from buffer. If error, return None.
    """

    try:
        response = client.hget(name=key, key=field)
    except ClientError as e:
        # AllAccessDisabled error == client lost
        logging.error(e)
        return None
    return response


def _hget_object_or_wait(client, key, field, sleep_time, time_out=60):
    """Retrieve an object from configured redis under specified key and field; wait if not exist

    :param client: redis client object
    :param key: string
    :param field: string
    :param sleep_time: float
    :param time_out: int (seconds)
    :return: direct bytestream, no need to read from buffer. If error, return None.
    """
    # Connect to redis
    # client = redis.Redis(host=endpoint, port=6379, db=0)
    # Retrieve the object

    try:
        start_time = time.time()
        while True:
            if time.time() - start_time > time_out:
                return None
            response = client.hget(name=key, key=field)
            if response != None:
                return response
            time.sleep(sleep_time)
    except ClientError as e:
        # AllAccessDisabled error == client lost
        logging.error(e)
        return None


def _list_keys(client, count=1000):
    """list the keys in configured redis

    :param client: redis client object
    :param count: maximum number of keys returned
    :return: List of keys in bytes. If error, return None. Maximum number of keys is 1000 in default node configuration.
    """

    try:
        response = client.scan(count=1000)[1]  # match allows for the pattern of keys
        names = response
    except ClientError as e:
        # AllAccessDisabled error == endpoint not found
        logging.error(e)
        return None
    if len(names) > 0:
        return names
    return None


def _hlist_keys(client, key, count=1000):
    """list all the fields within hash key in configured redis

    :param client: redis client object
    :param key: string
    :param count: maximum number of elements returned
    :return: list of fields in bytes, None if error. Maximum number of fields is 1000.
    """

    try:
        response = client.hscan(name=key, count=count)
        names = [*response[1]]
    except ClientError as e:
        # AllAccessDisabled error == endpoint or key not found
        logging.error(e)
        return None
    if len(names) > 0:
        return names
    return None


def _set_object(client, key, src_data):
    """Add value from configured redis under specified key

    :param client: redis client object
    :param key: string
    :param src_data: bytestream, int, string
    :return: True if src_data was added to redis at endpoint under key, otherwise False
    """

    if isinstance(src_data, bytes) or isinstance(src_data, str) or isinstance(src_data, int):
        object_data = src_data
    else:
        logging.error('Type of ' + str(type(src_data)) +
                      ' for the argument \'src_data\' is not supported.')

    try:
        response = client.set(name=key, value=object_data)
    except ClientError as e:
        # AllAccessDisabled error == client lost
        logging.error(e)
        return False
    return True


def _hset_object(client, key, field, src_data):
    """Add value from configured redis under specified key of certain hashtable

    :param client: redis client object
    :param field: string
    :param key: string
    :param src_data: bytestream, int, string
    :return: True if src_data was added to redis at endpoint under key, otherwise False
    """
    if isinstance(src_data, bytes) or isinstance(src_data, str) or isinstance(src_data, int):
        object_data = src_data
    else:
        logging.error('Type of ' + str(type(src_data)) +
                      ' for the argument \'src_data\' is not supported.')

    try:
        response = client.hset(name=key, key=field, value=object_data)
    except ClientError as e:
        # AllAccessDisabled error == client lost
        logging.error(e)
        return False
    return True


def _clear_all(client):
    """

    :param client: redis client object
    :return: True if flush_all successes, False otherwise
    """
    try:
        response = client.flushall()
    except ClientError as e:
        # AllAccessDisabled error == client lost
        logging.error(e)
        return False
    return True
