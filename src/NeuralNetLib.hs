module NeuralNetLib where 

import Neuron
import Utility
import Data.List

data NeuralNet a = NN { 
                      net :: [Layer (Neuron a)] }
                 | InducedNN {
                      net :: [Layer (Neuron a)],
                      output :: [a] } deriving (Show)


data Layer a = Layer {
                nodes :: [a]
             } deriving (Eq, Show)

instance Functor Layer where 
  fmap f (Layer xs) = Layer (map f xs)

----------------------------------------------------------------
---------------Views for NeuralNet Datatype---------------------
----------------------------------------------------------------
    
----------------------------------------------------------------
------------------- Forward Propogation ------------------------
----------------------------------------------------------------

induce_net :: (Num a) => ActivationFunction a -> NeuralNet a -> [a] -> (NeuralNet a)
induce_net af nn input = InducedNN in_net o 
  where layers = net nn 
        first_layer_induced = fmap (induce (1 : input)) (head layers)
        in_net = foldl (induce_layer af) [first_layer_induced] (tail layers)
        phi    = activation af
        o      = map phi (layer_ilf (last in_net))
        
clear_net :: NeuralNet a -> NeuralNet a 
clear_net nn = NN cleared_layers
  where layers = net nn 
        cleared_layers = map (fmap uninduce) layers

-- gives a layers ilf's if it is induced
layer_ilf :: (Num a) => Layer (Neuron a) -> [a]
layer_ilf layer = if allInduced 
                  then map local_field ns
                  else error "field is not induced"
  where ns = nodes layer
        allInduced = and $ map isInduced ns
  
                                                
induce_layer :: (Num a) => ActivationFunction a 
                       -> [Layer (Neuron a)]    --accumilator w/ induced layers
                       -> Layer (Neuron a)      --current layer to induce
                       -> [Layer (Neuron a)]    --accumilated output
induce_layer af induced current_layer = induced ++ [induced_current_layer]
  where phi = activation af
        previous_layer = last induced
        layer_input = map phi (layer_ilf previous_layer)
        induced_current_layer = fmap (induce layer_input) current_layer
        
induce_net' :: (Num a) => ActivationFunction a -> NeuralNet a -> [a] -> (NeuralNet a)
induce_net' af (NN (l:ls)) xs = InducedNN (l':ls') o
  where ys  = layer_outputs af l'
        l'  = Layer $ map (induce (1:xs)) (nodes l)
        (InducedNN ls' o) = induce_net' af (NN ls) ys 
induce_net' af (NN []) xs     = InducedNN [] xs
induce_net' _ _ _             = error "dont induce an induced net"

----------------------------------------------------------------
---------------------- Back Propogation ------------------------
----------------------------------------------------------------

data TrainingConfig a = TrainingConfig {
  act_func :: (ActivationFunction a),
  alpha :: a, 
  beta  :: a
} 

trainingConfig :: (Num a) => ActivationFunction a -> a -> a -> TrainingConfig a 
trainingConfig af a b = TrainingConfig af a b

newtype WeightChange a = WeightChange [[[a]]] deriving (Eq, Show)


backPropogate :: (Num a) => TrainingConfig a             -- config
                         -> ([a], [a])                   -- training_data@(input, desired outputs)
                         -> WeightChange a               -- last set of weight updates
                         -> NeuralNet a                  -- net to backprop
                         -> (NeuralNet a, WeightChange a)  -- updated net
                                                  
backPropogate config (xs, ds) dw nn = (NN (reverse ls), dws)
  where (InducedNN nss o) = induce_net (act_func config) nn xs
        reversed_net = InducedNN (reverse nss) o
        (NN ls, dws) = backPropLayers config ((1:xs),ds) dw [] [] reversed_net


backPropLayers :: (Num a) => TrainingConfig a                -- config
                          -> ([a],[a])                       -- training_data@(input, desired_output)
                          -> WeightChange a                  -- past iterations weight changes
                          -> [a]                             -- previous layers gradients
                          -> [[a]]                           -- previous layers weights in transpose form(outgoing from current layer)
                          -> NeuralNet a                     -- the induced net to evaluate
                          -> (NeuralNet a, WeightChange a)   -- updated net and weight deltas
backPropLayers config _ dw _ _ (InducedNN [] output) = (NN [], WeightChange [])
backPropLayers config t@(xs ,ds) dw pg plw (InducedNN (l:ls) output) = (resultNet, resultDeltaW)
  where resultNet    = NN (l':ls') 
        resultDeltaW = WeightChange (dw':dwss')
        ((NN ls'),(WeightChange dwss')) = backPropLayers config t (WeightChange dwsss_old) gradients ws nn' -- next recursion
                         where ws  = transpose $ layer_weights l
                               nn' = InducedNN (ls) output
        l'           = update_layer_weights dw' l 
        dw'          = layer_weight_deltas config dwss_old gradients ys
                         where ys = case ls of 
                                      [] -> xs 
                                      _  -> layer_outputs (act_func config) (head ls)
        gradients    = let errs = case (pg, plw) of 
                                    ([],[]) -> zipWith (-) ds output
                                    _       -> hidden_layer_errors pg plw
                       in layer_gradients (act_func config) l errs      
        (dwss_old:dwsss_old) = case dw of 
                                 WeightChange [] -> repeat $ repeat $ repeat 0
                                 WeightChange xs -> xs
{-                                 
backPropLayers _ _ _ [] [] (InducedNN (l:ls) output) = error "empty prev weight"   
backPropLayers config t dw pg plw (InducedNN (l:l_nxt:ls) output) = (resultNet, resultDeltaW)      
-}
  
update_layer_weights :: (Num a) => [[a]] -> Layer (Neuron a) -> Layer (Neuron a)
update_layer_weights wdss l = if (length wdss) == (layer_size l)
                              then Layer $ map Neuron $ zipWith (zipWith (+)) wss wdss
                              else error "layer and update sizes do not match"
  where wss = layer_weights l
        
layer_weight_deltas :: (Num a) => TrainingConfig a -> [[a]] -> [a] -> [a] -> [[a]]
layer_weight_deltas (TrainingConfig _ a b) dwss_old gradients ys = zipWith (zipWith (+)) momentums descents
  where momentums = map (map (* b)) dwss_old
        descents  = map (\g -> map (g *) ys) (map (* a) gradients)
        
layer_gradients :: (Num a) => ActivationFunction a -> Layer (Neuron a) -> [a] -> [a]
layer_gradients af l errs = zipWith (*) (map phi' ilfs) errs
  where phi' = derivitive af 
        ilfs = map (local_field) (nodes l)
        

hidden_layer_errors :: (Num a) => [a] -> [[a]] -> [a]
hidden_layer_errors pgs fwd_wss = map (sum . (zipWith (*) pgs)) fwd_wss

layer_outputs :: (Num a) => ActivationFunction a -> Layer (Neuron a) -> [a]
layer_outputs af (Layer ns) = map ((activation af) . local_field) ns 

layer_weights :: Layer (Neuron a) -> [[a]] 
layer_weights l = map weights (nodes l)

-- input is forward layer
-- output is list of forward weights for each neuron
layer_forward_weights :: (Num a) => Layer (Neuron a) -> [[a]]
layer_forward_weights = transpose . layer_weights

layer_size :: Layer a -> Int
layer_size (Layer xs) = length xs

----------------------------------------------------------------
---------------------- Back Propogation Train ------------------
----------------------------------------------------------------

train :: (Num a) => TrainingConfig a -> [([a], [a])] -> (NeuralNet a, WeightChange a) -> (NeuralNet a, WeightChange a)
train c (t:ts) (nn, dws) = train c ts $ backPropogate c t dws nn 
train c [] result = result


