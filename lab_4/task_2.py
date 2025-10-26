import time

def timeti(func):
    def logtim(n: int) -> int:
        start = time.time()
        res = func(n)
        print(f"Execution time: {time.time() - start}")
        return res
    return logtim

@timeti
def slow(n:int) -> int:
    time.sleep(0.05)
    return n*n

def main():
    print(slow(int(input())))

if __name__ == "__main__":
    main()