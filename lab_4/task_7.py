from typing import Iterable

def filter_emails(items: Iterable[str]) -> list[str]:
    return list(filter(lambda k: len(k.split("@")) > 1 and len(k.split(".")) > 1, items))
def main():
    emails = ["a@b.com", "wrong@", "x@y", "john.doe@mail.org"]
    print(filter_emails(emails))

if __name__ == "__main__":
    main()