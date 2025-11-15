import math
from dataclasses import dataclass

@dataclass(frozen=True)
class Complex:
    real: float
    imag: float

def add(a: Complex, b: Complex) -> Complex:
    match (a, b):
        case Complex(re1, im1), Complex(re2, im2):
            return Complex(re1 + re2, im1 + im2)

def difference(a: Complex, b:Complex) -> Complex:
    match (a, b):
        case Complex(re1, im1), Complex(re2, im2):
            return Complex(re1 - re2, im1 - im2)

def mul(a: Complex, b: Complex) -> Complex:
    match (a, b):
        case Complex(re1, im1) , Complex(re2, im2):
            return Complex(re1 * re2  - im1 * im2, re1 * im2 + im1 * re2)

def conjugate(a: Complex) -> Complex:
    match a:
        case Complex(re1, im1):
            return Complex(re1, -im1)

def module(a: Complex) -> float:
    match a:
        case Complex(re1, im1):
            return math.sqrt(re1**2+im1**2)


def main():
    a, b = list(map(float, input("Enter real, imag: ").split()))
    z = Complex(a, b)
    a, b = list(map(float, input("Enter real, imag: ").split()))
    z1 = Complex(a, b)
    print("Add: ", add(z, z1))
    print("Difference: ", difference(z, z1))
    print("Mul: ", mul(z, z1))
    print("Conjugate: ", conjugate(z))
    print("Module: ", module(z))

if __name__ == "__main__":
    main()