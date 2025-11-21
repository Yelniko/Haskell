def name_score_pair(name: list[str], scores: dict[str, int]):
    for i in name:
        if i in scores:
            yield (i, scores[i])

def main():
    names = ["Іван", "Варя", "Ольга", "Богдан"]
    scores = {"Іван": 90, "Ольга": 85, "Богдан": 78}

    it = name_score_pair(names, scores)

    for i in it:
        print(i, end= " ")

if __name__ == "__main__":
    main()