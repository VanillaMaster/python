from multiprocessing import Process, Queue, Manager
import time


def f(q,b):
    while (b.value or (not q.empty())):
        while (not q.empty()):
            f = open("demofile2.txt", "a")
            f.write(q.get())
            f.close()
        time.sleep(1)

if __name__ == '__main__':
    q = Queue()
    check = Manager().Value('i',1)
    p = Process(target=f, args=(q,check))
    p.start()


    q.put("1 ")
    q.put("5 ")
    q.put("4 ")
    q.put("8 ")

    print("main process end")

    check.value=0
    p.join()
    print("sub process end")
