
processList :: [Int] -> [Int]
processList [] = []
processList lis = take 3 ( map(\i -> i * i) (filter (\i -> i `mod` 2 /= 0 ) lis ) )

main = do
    print ( processList [1..10] )