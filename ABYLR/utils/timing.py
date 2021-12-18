from logging import log
import timeit
from .logging import logger


def timecal(func):
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        ret = func(*args, **kwargs)
        end = timeit.default_timer()
        # print('Time taken: ', end - start)
        logger.info(f"Time cost: {end - start:.3f}")
        return ret
    return wrapper
