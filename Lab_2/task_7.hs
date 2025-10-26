

avg [] = Nothing
avg lis = Just ( sum lis / fromIntegral ( length lis) )

-- Реалізуйте функцію avg, яка обчислює середнє значення списку. Якщо список порожній, повертає Nothing

main = do
    number_lis <- map read . words <$> getLine
    print ( avg number_lis )