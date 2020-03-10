#making sure to empty the bucket before start.


def main(event, context):
    
    storage = event['storage']
    if storage == "memcache":
        from elasticache.Memcache.__init__ import memcache_init
        from elasticache.Memcache.clear_all import clear_all

        memcache_location = event['elasticache']
        endpoint = memcache_init(redis_location)
        
        clear_all(endpoint)
    if storage == "redis":
        from elasticache.Redis.__init__ import redis_init
        from sync.sync_grad_redis import clear_bucket
        
        redis_location = event['elasticache']
        grad_bucket = event['grad_bucket']
        model_bucket = event['model_bucket']
        endpoint = redis_init(redis_location)
    
        clear_bucket(endpoint,grad_bucket)
        clear_bucket(endpoint,model_bucket)