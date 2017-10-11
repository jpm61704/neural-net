module Utility where 

import Control.Exception

data ActivationFunction a = Activation {activation :: (a -> a) , derivitive :: (a -> a)}

logistic_function :: Double -> (Double -> Double)
logistic_function a x = 1 / (1 + (e ** ((-a) * x)))

logistic_derivitive :: Double -> (Double -> Double)
logistic_derivitive a x = (a * (e ** (a * x))) / ((e ** (a * x) + 1) ** 2)

logisticActivation :: Double -> ActivationFunction Double
logisticActivation a = Activation (logistic_function a) (logistic_derivitive a)

dot :: (Num a) => [a] -> [a] -> a 
dot xs ys = assert ((length xs) == (length ys)) (sum (zipWith (*) xs ys))


e :: Double
e = 2.71828182845904523


        
        
        