def append_tuple(tpl, item):
    return tpl + (item,)

def main():
    lis = tuple(input().split())
    element = input()

    t = append_tuple(lis, element)

    print(lis)
    print(t)

if __name__ == "__main__":
    main()