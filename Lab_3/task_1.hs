
updateAt::Int -> a -> [a] -> [a]
updateAt index element [] = []
updateAt index element lis = take index lis ++ [element] ++ drop (index+1) lis

main = do
    print(updateAt (-1) (-7) [1, 3, 2])