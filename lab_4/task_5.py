from typing import Iterable

def has_prime(nums: Iterable[int]) -> bool:
    return any(map(lambda e: all(e % i for i in range(2, int(e**0.5 + 1))) if e > 1 else False, nums))

def main():
    numbers = list(map(int, input().split()))
    print(has_prime(numbers))

if __name__ == "__main__":
    main()