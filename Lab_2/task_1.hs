
safeHead :: [a] -> Maybe a
safeHead [] = Nothing
safeHead (x:_) = Just x

main = do
    number_lis <- words <$> getLine
    print ( safeHead number_lis )