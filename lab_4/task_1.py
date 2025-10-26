from functools import wraps

def log_calls(func):
    @wraps(func)
    def log(*args) -> int:
        print(f"Name fan: {func.__name__}\nArguments: {args}")
        res = func(args[0], args[1])
        print(f"Result: {res}")
        return res

    return log

@log_calls
def add(a: int, b: int) -> int:
    return a + b

def main():
    number = list(map(int, input().split()))
    add(number[0], number[1])

if __name__ == "__main__":
    main()