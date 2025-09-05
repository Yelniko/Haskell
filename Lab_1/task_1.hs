
doubleEven :: [Int] -> [Int]
doubleEven x = map (\i -> if i `mod` 2 == 0 then i*2 else i) x

main::IO()
main = do
    number <- map read . words <$> getLine
    print ( doubleEven number )