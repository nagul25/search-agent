import os
import time
from fastapi import Request
from fastapi.responses import JSONResponse
import redis
from app.log_config import logger

# Allow overriding Redis host/port for containerized deployments
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
window_size = 60  # time window in seconds
max_requests = 60  # maximum number of requests allowed within the window


async def rate_limiter(request: Request, call_next):
    client_id = request.client.host
    logger.info(f"Rate limiting check for client: {client_id}")
    
    current_window  = int(time.time() // window_size)

    key = f"rate_limiter:{client_id}:{current_window}"
    # check if redis exists, increment the count
    # if redis not exists, then allow the request 
    try:
        current_count = r.incr(key)
    except redis.exceptions.RedisError as e:
        logger.error(f"Redis error: {e}")
        return await call_next(request)
    

    if current_count == 1:
        r.expire(key, window_size)

    if current_count > max_requests:
        logger.warning(f"Rate limit exceeded for client: {client_id}")
        return JSONResponse(status_code=429, content={"detail": "Too Many Requests"})

    response = await call_next(request)
    return response