
findIndex _ [] = Nothing
findIndex i (el:lis)
    | el == i  = Just 0
    | otherwise =
        case findIndex i lis of
            Nothing -> Nothing
            Just n -> Just ( n + 1)

-- Напишіть функцію findIndex, яка повертає індекс першого входження елемента в списку, або Nothing, якщо елемент не знайдено.

main = do
    print ( findIndex 3 [1, 2, 3, 4] )
    print ( findIndex 9 [1, 2, 3] )
    print ( findIndex 1 [1, 2, 3, -4, -1, -4, 1] )


