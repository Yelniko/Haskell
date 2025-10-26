
type Move = (Char, Char)
hanoi :: Int -> Char -> Char -> Char -> [Move]
hanoi 0 _ _ _ = []
hanoi n from aux to = do
    hanoi (n - 1) from to aux
    ++ [(from, to)]
    ++ hanoi (n - 1) aux from to



main = do
    print ( hanoi 5 'A' 'B' 'C' )
    --print ( hanoi 3 'L' 'M' 'R' )