def substrings(string: str, number: int):
    if len(string) > number:
        start = 0
        stop = number
        for i in range(len(string) - number + 1):
            start += 1
            stop += 1
            yield string[start - 1:stop - 1]


def main():
    s = substrings("abcdef", 3)
    for i in s:
        print(i)

if __name__ == "__main__":
    main()