import logging
import memcache
from botocore.exceptions import ClientError




def hlist_keys(client, keys):
    #client = memcache.Client([client+":"+str(11211)])
    #client = client
    objects = client.get_multi(keys)
    #print(objects)
    if bool(objects):
        return objects
    else:
        return None

def handler(event,context):
    #mc = memcache.Client(['convergence.fifamc.cfg.euc1.cache.amazonaws.com:11211'])
    mc = 'convergence.fifamc.cfg.euc1.cache.amazonaws.com'
    print(hlist_keys(mc,["tmp_value_g_1","tmp_value_g_2"]))
