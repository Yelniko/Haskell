
absList :: [Int] -> [Int]
absList x = map (\i -> if i < 0 then i * (-1) else i) x

main :: IO()
main = do
    number <- map read . words <$> getLine
    print (absList number)