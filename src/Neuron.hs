module Neuron where 

import Utility

data Neuron a = Neuron {
                weights :: [a] }  
              | InducedNeuron{
                weights :: [a],
                local_field :: a
              }
              deriving (Show)

instance Functor Neuron where 
  fmap f (Neuron ws) = Neuron $ (map f ws)
  

type Bias = Weight
type Weight = Double
type Inputs = [Double]
type Output = Double

induce :: (Num a) => [a] -> Neuron a -> Neuron a 
induce input neuron = InducedNeuron (weights neuron) (induced_local_field neuron input)

uninduce :: Neuron a -> Neuron a 
uninduce n = Neuron (weights n)

isInduced :: Neuron a -> Bool
isInduced (Neuron _) = False 
isInduced (InducedNeuron _ _) = True

induced_local_field :: (Num a) => Neuron a -> [a] -> a 
induced_local_field (Neuron w) i = dot i w

output_signal :: (Num a) => Neuron a -> (a -> a) -> [a] -> a 
output_signal n af i = (af . (induced_local_field n)) i
                                                        --desired signal
error_signal :: (Num a) => Neuron a -> (a -> a) -> [a] -> a -> a
error_signal n af x d = d - y
    where y = output_signal n af x