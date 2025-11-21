def flatten(lis: list):
    for i in lis:
        if type(i) == list:
            yield from flatten(i)
        else:
            yield i

def main():
    data = [1, [2, 3], [4, [5, 6]], 7]
    test = flatten(data)

    for i in test:
        print(i, end = " ")

if __name__ == "__main__":
    main()