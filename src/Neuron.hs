module Neuron where 

import Utility

data Neuron a = Neuron [a] deriving (Show)

instance Functor Neuron where 
  fmap f (Neuron ws) = Neuron $ (map f ws)
  
data InducedNeuron a = InducedNeuron {
  local_field :: a, 
  weights :: [a]
}

inducedNeuron :: (Num a) => (Neuron a) -> a -> (InducedNeuron a) 
inducedNeuron (Neuron w) ilf = InducedNeuron ilf w

type Bias = Weight
type Weight = Double
type Inputs = [Double]
type Output = Double

induced_local_field :: (Num a) => Neuron a -> [a] -> a 
induced_local_field (Neuron w) i = dot i w


output_signal :: (Num a) => Neuron a -> (a -> a) -> [a] -> a 
output_signal n af i = (af . (induced_local_field n)) i

                                                        --desired signal
error_signal :: (Num a) => Neuron a -> (a -> a) -> [a] -> a -> a
error_signal n af x d = d - y
    where y = output_signal n af x