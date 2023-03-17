import time

@profile
def heavy_loop():
    for i in range(5000000):
        loop10()

    for i in range(10000):
        loop10()

    for i in range(10):
        time.sleep(0.1)

    print("executed loop")

def loop10():
    for i in range(10):
        pass

heavy_loop()