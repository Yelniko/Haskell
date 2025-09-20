
binarySearch :: Ord a => a -> [a] -> Maybe Int
binarySearch x xs = go 0 (length xs - 1)
  where
    go lo hi
      | lo > hi   = Nothing
      | otherwise =
          let mid = (lo + hi) `div` 2
              val = xs !! mid
          in case compare x val of
              LT -> go lo (mid - 1)
              GT -> go (mid + 1) hi
              EQ -> Just mid

main = do
    print ( binarySearch 5 [1,3,5,7,9] )
    print ( binarySearch 4 [1,3,5,7,9] )
    print ( binarySearch 10 [] )