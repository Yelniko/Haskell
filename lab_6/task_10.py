def running_status(numbers: list[float]):
    number = 0
    for i in range(len(numbers)):
        number += numbers[i]
        yield (i+1, numbers[i], number/(i+1))

def main():
   it = running_status([10, 20, 30])

   for i in it:
       print(i, end = " ")

if __name__ == "__main__":
    main()