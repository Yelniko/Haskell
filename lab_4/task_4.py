from typing import Iterable

def all_close(xs: Iterable[float], tol: float = 1e-6) -> bool:
    return all(list(map(lambda k: abs(xs[0] - k)  < tol, xs)))

def main():
    print(all_close([1.0, 1.0000005, 0.9999999], tol=1e-5))
    print(all_close([1.0, 1.1], tol=1e-3))

if __name__ == "__main__":
    main()