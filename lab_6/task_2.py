class SkipIterator:
    def __init__(self, lis: list, number: int):
        self.lis = lis
        self.number = number
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        while (True):
            if self.index < len(self.lis):
                if self.index % self.number == 0:
                    self.index += 1
                    return self.lis[self.index - 1]
            else:
                raise StopIteration
            self.index += 1



def main():
    lis = input("Enter list: ").split()
    number = int(input("Enter number: "))
    it = SkipIterator(lis, number)

    for i in it:
        print(i, end = " ")

if __name__ == "__main__":
    main()