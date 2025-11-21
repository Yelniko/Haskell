def tuples_to_dict(pairs: list[tuple]) -> dict:
    d = {}
    for i in pairs:
        d[i[0]] = i[1]
    return d

def dict_to_tuples(d: dict) -> list[tuple]:
    lis = []
    for i, j in d.items():
        lis.append((i, j))
    return lis

def main():
    t = dict_to_tuples({"h": 1, "t":2 })
    print(t)
    print(tuples_to_dict(t))

if __name__ == "__main__":
    main()