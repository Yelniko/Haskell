def even_in_range(start: int, end: int):
    for i in range(start, end + 1):
        if i % 2 == 0:
            yield i

def main():
    lis_con = list(map(int, input("Enter start, end: ").split()))
    number_gen = even_in_range(lis_con[0], lis_con[-1])
    lis = list()

    for i in number_gen:
        lis.append(i)

    print(lis)

if __name__ == "__main__":
    main()