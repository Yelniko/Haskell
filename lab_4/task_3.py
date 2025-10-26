import re

def is_strong_password(s: str) -> bool:
    return True if len(s) >= 8 and re.search(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()])', s) else False

def main():
    s = input()
    print(is_strong_password(s))

if __name__ == "__main__":
    main()