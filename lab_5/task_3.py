from dataclasses import dataclass

@dataclass(frozen=True)
class Matrix:
    m11: float
    m12: float
    m21: float
    m22: float

def det(m: Matrix) -> float:
    match m:
        case Matrix(m11, m12, m21, m22):
            return m11 * m22 - m12 * m21

def rev(m: Matrix):
    match m:
        case Matrix(m11, m12, m21, m22):
            d = det(m)
            return "Not" if d == 0 else Matrix(m22/d, -m12/d, -m21/d, m11/d)

def bod(a: Matrix, b:Matrix):
    match (a, b):
        case Matrix(a11, a12, a21, a22), Matrix(b11, b12, b21, b22):
            return Matrix(
                a11*b11+a12*b21,
                a11*b12+a12*b22,
                a21*b11+a22*b21,
                a21*b12+a22*b22
            )

def main():
    lis = list(map(float, input().split()))
    m1 = Matrix(lis[0], lis[1], lis[2], lis[3])
    lis = list(map(float, input().split()))
    m2 = Matrix(lis[0], lis[1], lis[2], lis[3])

    print(det(m1))
    print(rev(m1))
    print(bod(m1, m2))

if __name__ == "__main__":
    main()