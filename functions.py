from time import time
def timer(func):
    def time_func(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f'Function {func.__name__!r} executed in {(end-start):.2f}s')
        return result
    return time_func