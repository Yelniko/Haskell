

avg [] = Nothing
avg lis = Just ( sum lis / fromIntegral ( length lis) )

main = do
    number_lis <- map read . words <$> getLine
    print ( avg number_lis )