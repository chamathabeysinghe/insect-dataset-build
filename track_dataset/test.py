from multiprocessing.pool import ThreadPool


def hello(i):
    print("{} \n".format(i))

p = ThreadPool(2)
p.map(hello, range(3))