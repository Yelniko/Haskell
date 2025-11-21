def aggregate_pairs(pairs: list[tuple[str, int]]) -> dict[str, int]:
    d = dict()
    for i in pairs:
        if i[0] in d:
            d[i[0]] += i[1]
        else:
            d[i[0]] = i[1]
    return d

def main():
    pairs = [("apple", 3), ("banana", 5), ("apple", 2), ("orange", 4)]
    print(aggregate_pairs(pairs))

if __name__ == "__main__":
    main()