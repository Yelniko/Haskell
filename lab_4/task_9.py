def top_k(students: list[dict[str, int | str]], k: int) -> list[str]:
    students.sort(key = lambda k: k['score'])
    students = students[::-1][:k]
    return [i["name"] for i in students]

def main():
    students = [{"name": "Ann", "score": 90}, {"name": "Bob", "score": 95}, {"name": "Ada", "score": 95}]
    print(top_k(students, 2))


if __name__ == "__main__":
    main()
