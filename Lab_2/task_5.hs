
myMap :: (a -> b) -> [a] -> [b]
myMap f [] = []
myMap f (el:lis) = f el : myMap f lis

main = do
    print( myMap (*2) [1, 2, 3] )
    print( myMap show [1, 2, 3])