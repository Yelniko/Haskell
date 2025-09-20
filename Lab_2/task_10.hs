

applyAll [] i = i
applyAll (f:lis_f) i = applyAll lis_f (f i)

main = do
    print ( applyAll [(*2), (+1), (^2)] 3)
    print ( applyAll [ negate, abs ] (-5))

