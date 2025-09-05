
myReverse :: [a] -> [a]
myReverse [] = []
myReverse (x:lis) = myReverse lis ++ [x]

main :: IO()
main = do
    number <- words <$> getLine
    print (myReverse number)