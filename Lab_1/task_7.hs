
main :: IO()
main = do
    print ([(a, d, c) | a <- [1..20], d <- [1..20], c <-[1..20], a^2+d^2==c^2, a <= d])