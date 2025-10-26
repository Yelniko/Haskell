def minmax_scale(xs: list[float]) -> list[float]:
    min_e, max_e = min(xs), max(xs)
    return list(map(lambda e: (e - min_e) / (max_e - min_e), xs)) if max_e != min_e else list(map(lambda e: 0.0, xs))

def main():
    number = list(map(int, input().split()))
    print(minmax_scale(number))

if __name__ == "__main__":
    main()