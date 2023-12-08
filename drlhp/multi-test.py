from multiprocessing import Process


class Obj:
    pass


def main():
    o = Obj()
    test = []

    def child():
        print(f"child obj: {repr(o)}")
        test.append(1)

    p = Process(target=child, daemon=True)
    p.start()
    print(f"parent obj: {repr(o)}")
    p.join()
    print(test)


main()
