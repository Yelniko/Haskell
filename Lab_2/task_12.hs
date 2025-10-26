

quickSort :: [Int] -> [Int]
quickSort [] = []
quickSort (x:xs) =
    let smaler  = [y | y <- xs, y <= x]
        greter  = [y | y <- xs, y >  x]
    in quickSort smaler ++ [x] ++ quickSort greter

main = do
    print ( quickSort [3,1,4,1,5,9,2] )
    print ( quickSort [1, -3] )
    print ( quickSort [] )