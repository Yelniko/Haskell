
safeDiv (_:0:_) = Nothing
safeDiv (a:d:_) = Just (a / d)

main = do
    number_lis <- map read . words <$> getLine
    print ( safeDiv number_lis )