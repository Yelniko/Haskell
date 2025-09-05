
fid :: Int -> Int -> Int -> Int
fid 0 1 (-1) = 0
fid x y z = if z > 0 then fid y (x+y) (z-1) else y

main :: IO()
main = do
    number <- readLn
    print(fid 0 1 (number-1))