


mkMultiplier :: Int -> (Int -> Int)
mkMultiplier k = \x -> k * x

-- Напишіть функцію mkMultiplier, яка приймає число і повертає функцію множення на це число.

main  = do
    number_str <- getLine
    let number = read number_str :: Int
    let fan = mkMultiplier number
    number_1_str <- getLine
    let number_1 = read number_1_str :: Int
    print ( fan number_1 )