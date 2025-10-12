def push(stack: list, item) -> list:
    return [item] + stack

def main():
    lis = list(input().split())
    item = input()
    l = push(lis, item)
    print(lis)
    print(l)

if __name__ == "__main__":
    main()