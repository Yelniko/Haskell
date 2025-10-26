
removeAt::Int->[a]->[a]
removeAt index lis = take index lis ++ drop (index+1) lis

main = do
    print( removeAt 1 [10, 20, 30])
    print( removeAt 5 [1, 2, 3])
    print( removeAt 2 ["test_0", "test_1", "test_2"])