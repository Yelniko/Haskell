
sumOdd :: [Int] -> Int
sumOdd x = foldl (\i j -> if j `mod` 2 == 0 then j else i + j ) 0 x

sumOdd_2 :: [Int] -> Int
sumOdd_2 x = sum(filter (\i -> i `mod` 2 /= 0) x)

main::IO()
main = do
    numbers <- map read . words <$> getLine
    print (sumOdd numbers)
    print (sumOdd numbers)
