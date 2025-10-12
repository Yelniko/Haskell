
data Person = Person {name::String, age::Int} deriving Show
incrementAge::Person -> Person
incrementAge n = Person (name n) (age n + 1)

main = do
    let p = Person "Anna" 25
    print(incrementAge p)
    print(p)
