import logging
from functools import wraps
from time import time


#decorator que mede o tempo da execucao
# eh dinamico e recebe e imprime o nome da funcao executada, tal como 'treinamento'
def timeit(log_level=logging.INFO, alternative_title=None):
    def wrap(f):
        @wraps(f)  # keeps the f.__name__ outside the wrapper
        def wrapped_f(*args, **kwargs):
            t0 = time()
            result = f(*args, **kwargs)
            ts = round(time() - t0, 3)

            title = alternative_title or f.__name__
            logging.getLogger().log(
                log_level, " %s took: %f seconds" % (title, ts))

            return result

        return wrapped_f
    return wrap
