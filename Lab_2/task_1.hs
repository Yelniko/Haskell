
safeHead :: [a] -> Maybe a
safeHead [] = Nothing
safeHead (x:_) = Just x

-- Напишіть функцію safeHead, яка повертає перший елемент списку або Nothing, якщо список порожній.

main = do
    number_lis <- words <$> getLine
    print ( safeHead number_lis )