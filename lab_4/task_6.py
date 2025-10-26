from typing import Iterable

def scale_and_shift(xs: Iterable[float], scale: float, shift: float) -> list[float]:
    return list(map(lambda k: scale*k+shift, xs))


def main():
    numbers = list(map(int, input().split()))
    scale = int(input())
    shift = int(input())
    print(scale_and_shift(numbers, scale, shift))

if __name__ == "__main__":
    main()