
findIndex _ [] = Nothing
findIndex i (el:lis)
    | el == i  = Just 0
    | otherwise =
        case findIndex i lis of
            Nothing -> Nothing
            Just n -> Just ( n + 1)

main = do
    print ( findIndex 3 [1, 2, 3, 4] )
    print ( findIndex 9 [1, 2, 3] )


