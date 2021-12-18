import timeit


def timecal(func):
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        ret = func(*args, **kwargs)
        end = timeit.default_timer()
        print('Time taken: ', end - start)
        return ret
    return wrapper


def add(a, b):
    return a+b


def fuckyou():
    print("Jee")
    pass


fuck = timecal(add)(1, 2)
print(fuck)

timecal(fuckyou)()
