
myLength :: [a] -> Int
myLength [] = 0
myLength (_:lis) = 1 + myLength lis

main :: IO()
main = do
    number <- words <$> getLine
    print ( myLength number )
