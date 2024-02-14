import signal

from typing import Callable


def timeout(seconds: int, use_timeout: bool):
    def decorator(function: Callable):
        if not use_timeout:
            return function

        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f'This call took longer than {seconds} seconds')

            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            result = function(*args, **kwargs)
            signal.alarm(0)
            return result
        return wrapper
    return decorator
