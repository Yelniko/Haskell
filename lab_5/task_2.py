import math
from dataclasses import dataclass

@dataclass(frozen=True)
class Vector:
    x: float
    y: float

def norm(a:Vector) -> Vector:
    match a:
        case Vector(x, y):
            length = math.sqrt(x**2+y**2)
            return Vector(x / length, y / length)

def scal(a:Vector, b:Vector) -> float:
    match (a, b):
        case Vector(x1, y1), Vector(x2, y2):
            return x1 * x2 + y1 * y2

def vec(a:Vector, b:Vector) -> float:
    match (a, b):
        case Vector(x1, y1), Vector(x2, y2):
            return x1 * y2 - y1 * x2

def main():
    x, y = list(map(float, input("Enter x, y: ").split()))
    z = Vector(x, y)
    x, y = list(map(float, input("Enter x, y: ").split()))
    z1 = Vector(x, y)

    print("Norm: ", norm(z))
    print("Scal: ", scal(z, z1))
    print("Vec: ", vec(z, z1))

if __name__ == "__main__":
    main()