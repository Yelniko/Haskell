def update_dict(original: dict, key, value) -> dict:
    new_dict = original.copy()
    new_dict[key] = value
    return new_dict

def main():
    lis_key = input().split()
    lis_value = input().split()
    original = {}
    for i in range(len(lis_key)):
        original[lis_key[i]] = lis_value[i]

    key = input()
    value = input()

    dict_copy = update_dict(original, key, value)
    print(f"Original: {original}")
    print(f"Copy: {dict_copy}")


if __name__ == "__main__":
    main()