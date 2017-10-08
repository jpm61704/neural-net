module NeuralNetLib where 

import Neuron
import Utility

data NeuralNet a = NN [[[a]]] deriving(Eq, Show)

data Layer a = Output {
                nodes :: [a]
             }
             | Hidden {
                nodes :: [a]
             }
             deriving (Eq, Show)

instance Functor Layer where 
  fmap f (Output xs) = Output (map f xs)
  fmap f (Hidden xs) = Hidden (map f xs)
                      

----------------------------------------------------------------
---------------Views for NeuralNet Datatype---------------------
----------------------------------------------------------------
data NeuralNetNeuronView a = NNNeuronView [Layer (Neuron a)]

neuralNetNeuronView :: NeuralNet a -> NeuralNetNeuronView a
neuralNetNeuronView nn = 
  case neuralNetLayerView nn of
    NNLayerView layers -> NNNeuronView $ map (fmap Neuron) layers
    
data NeuralNetLayerView a = NNLayerView [Layer a]

neuralNetLayerView :: NeuralNet a -> NeuralNetLayerView [a]
neuralNetLayerView (NN xsss) = NNLayerView $ hiddenLayers ++ [outputLayer] 
  where hiddenLayers = map (Hidden) (init xsss) 
        outputLayer = Output $ last xsss
        
----------------------------------------------------------------
------------------- Forward Propogation ------------------------
----------------------------------------------------------------

forwardComputation :: (Num a) => ActivationFunction a -> NeuralNet a -> [a] -> ([a], [[a]])
forwardComputation af nn xs = case neuralNetNeuronView nn of
  NNNeuronView layers -> foldl (computeLayer af) (xs, []) (layers)
  
forwardPropogate :: (Num a) => ActivationFunction a -> NeuralNet a -> [a] -> [a]
forwardPropogate af nn = fst . forwardComputation af nn

computeLayer :: (Num a) => ActivationFunction a -> ([a], [[a]]) -> Layer (Neuron a) -> ([a],[[a]])
computeLayer af (xs, ilfs) layer = (y, ilfs ++ [ilf])
  where ilf = map (`induced_local_field` xs) (nodes layer) 
        y   = map (activation af) ilf

----------------------------------------------------------------
---------------------- Back Propogation ------------------------
----------------------------------------------------------------

findGradients :: (Num a) => ActivationFunction a -> NeuralNet a -> [[a]] -> [a] -> [[a]]
findGradients af nn ilfs err = case neuralNetLayerView of 
  NNLayerView layers -> foldr computeLayerGradients [] layers

computeLayerGradients :: (Num a) => Layer ([a], a) -> acc
computeLayerGradients (Hidden wss) grads = undefined
computeLayerGradients (Output wss) grads = undefined 
          

          
-- I believe that I need to save the previous layer in a tuple during this fold in order to see one layer forward

