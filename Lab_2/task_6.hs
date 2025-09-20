

myFilter :: (a -> Bool) -> [a] -> [a]
myFilter f [] = []
myFilter f (el:lis)
    | f el = el : myFilter f lis
    | otherwise = filter f lis

main = do
    print ( myFilter even [1..10] )
    print ( myFilter (>3) [1..5] )
