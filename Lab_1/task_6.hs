
myMaximum :: [Int] -> Int
myMaximum [x] = x
myMaximum (x:lis) = max x (myMaximum lis)

main :: IO()
main = do
    number_list <- map read . words <$> getLine
    print (myMaximum number_list)
