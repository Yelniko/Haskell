def filter_dict(d: dict[str, int], number: int)-> dict[str, int]:
    di = {}
    for i, j in d.items():
        if j > number:
            di[i] = j
    return di

def main():
    products = {"apple": 25, "banana": 40, "orange": 30, "kiwi": 55}
    print(filter_dict(products, 39))

if __name__ == "__main__":
    main()