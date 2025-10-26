from functools import reduce


def char_hist(s: str) -> dict[str, int]:
    return reduce(lambda k, e: {**k, e: k.get(e, 0) + 1}, s, {})

def main():
    print(char_hist(input()))

if __name__ == "__main__":
    main()