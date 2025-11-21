class KeyFilterIterator:
    def __init__(self, d: dict, allowed_keys):
        self.d = d
        self.allowed_keys = allowed_keys

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if len(self.allowed_keys) > 0:
                odj = self.allowed_keys.pop()
                if odj in self.d:
                    return (odj, self.d[odj])
            else:
                raise StopIteration


def main():
    d = {"a": 1, "b": 2, "c": 3}
    allowed = {"b", "c", "x"}

    it = KeyFilterIterator(d, allowed)

    for i in it:
        print(i, end = "")

if __name__ == "__main__":
    main()