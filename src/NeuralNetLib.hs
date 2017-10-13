module NeuralNetLib where 

import Neuron
import Utility
import Data.List

data NeuralNet a = NN { 
                      net :: [Layer (Neuron a)] }
                 | InducedNN {
                      net :: [Layer (Neuron a)],
                      output :: [a] } deriving (Show)

----------------------------------------------------------------
---------------Views for NeuralNet Datatype---------------------
----------------------------------------------------------------

data NetBackView a = NetBackView (NeuralNet a) deriving (Show)

netBackView :: NeuralNet a -> NetBackView a 
netBackView (NN ls) = NetBackView $ NN $ reverse ls 
netBackView (InducedNN ls o) = NetBackView $ InducedNN (reverse ls) o

----------------------------------------------------------------
---------------------------Layers-------------------------------
----------------------------------------------------------------

data Layer a = Layer {
                nodes :: [a]
             } deriving (Eq, Show)

instance Functor Layer where 
  fmap f (Layer xs) = Layer (map f xs)
  
layer_size :: Layer a -> Int
layer_size (Layer xs) = length xs

      ------------------------------------------------
      -------------------Neuron Layers----------------
      ------------------------------------------------


layer_outputs :: (Num a) => ActivationFunction a -> Layer (Neuron a) -> [a]
layer_outputs af (Layer ns) = map ((activation af) . local_field) ns 

layer_weights :: Layer (Neuron a) -> [[a]] 
layer_weights l = map weights (nodes l)

-- input is forward layer
-- output is list of forward weights for each neuron
layer_forward_weights :: (Num a) => Layer (Neuron a) -> [[a]]
layer_forward_weights = tail . transpose . layer_weights

-- correct
update_layer_weights :: (Num a) => Layer [a] -> Layer (Neuron a) -> Layer (Neuron a)
update_layer_weights (Layer weight_changes) l
  | sizes_are_equal = Layer $ map Neuron $ zipWith (zipWith (+)) wss weight_changes
  | otherwise       = error "layer and update sizes do not match"
  where sizes_are_equal = (length weight_changes) == (layer_size l)
        wss = layer_weights l
    
----------------------------------------------------------------
------------------- Forward Propogation ------------------------
----------------------------------------------------------------
        
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
  
                                                
        
induce_net :: (Num a) => ActivationFunction a -> NeuralNet a -> [a] -> (NeuralNet a)
induce_net af (NN (l:ls)) xs = InducedNN (l':ls') o
  where ys  = layer_outputs af l'
        l'  = Layer $ map (induce (1:xs)) (nodes l)
        (InducedNN ls' o) = induce_net af (NN ls) ys 
induce_net af (NN []) xs     = InducedNN [] xs
induce_net _ _ _             = error "dont induce an induced net"

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

newtype WeightChange a = WeightChange [Layer [a]] deriving (Eq, Show)


-- THIS FUNCTION IS CORRECT on net1 example!!!
back_propogate_output :: (Num a) => TrainingConfig a      -- config
                                 -> [a]                   -- desired output
                                 -> Layer [a]             -- previous backprop changes for output layer 
                                 -> Layer (Neuron a)      -- backward layer for outputs
                                 -> Layer (Neuron a)      -- output layer 
                                 -> (Layer (Neuron a), Layer [a], Layer a, Layer (Neuron a))
back_propogate_output cfg@(TrainingConfig af a b) ds momentum_layer next_layer current_layer = (updated_output_layer, weight_changes, layer_grads, current_layer)
  where updated_output_layer = update_layer_weights weight_changes current_layer 
        weight_changes = weight_change cfg momentum_layer layer_grads (Layer (1:(layer_outputs af next_layer))) -- !! Change made here
        layer_grads = Layer $ layer_gradients af current_layer errors
          where errors = let os = layer_outputs af current_layer in zipWith (-) ds os 



-- correctedness == unknown
back_propogate_hidden :: (Num a) => TrainingConfig a          -- config                                    
                                 -> [a]                       -- input                          
                                 -> [Layer [a]]               -- previous backprop changes for (layer:net)
                                 -> NeuralNet a               -- rest of net
                                 -> ([Layer (Neuron a)],      -- accumilated layers 
                                     [Layer [a]],             -- accumilated weight changes 
                                     Layer a,                 -- forward layer's gradients
                                     Layer (Neuron a))        -- forward layer's weights
                                 -> ([Layer (Neuron a)], [Layer [a]], Layer a, Layer (Neuron a))
back_propogate_hidden _ _ _ (NN []) (acc_ls, acc_dwsss, fwd_grads, fwd_wss) = (acc_ls, acc_dwsss, fwd_grads, fwd_wss)   
back_propogate_hidden config xs [] nn (acc_ls, acc_dwsss, fwd_grads, fwd_wss) = error "previous backprop changes are empty"
back_propogate_hidden config@(TrainingConfig af _ _) xs (dwss_l:dwss_ls) nn (acc_ls, acc_dwsss, (Layer fwd_grads), fwd_wss) = 
    back_propogate_hidden config xs dwss_ls (NN ls) (acc_ls', acc_dwsss', layer_grads, layer)
    where (layer:ls) = net nn
          acc_ls'    = updated_layer : acc_ls 
          acc_dwsss' = weight_changes : acc_dwsss
          updated_layer = update_layer_weights weight_changes layer  
          weight_changes = weight_change config dwss_l layer_grads ys
            where ys = case ls of
                       (back:_) -> (Layer (1 : (layer_outputs af back))) -- !!! made change here 
                       []       -> Layer xs
          layer_grads = Layer $ layer_gradients af layer errors 
            where errors = map sum $ map (zipWith (*) (fwd_grads)) (layer_forward_weights fwd_wss)

-- correctedness == unknown
backPropogate :: (Num a) => TrainingConfig a 
                         -> ([a],[a])
                         -> WeightChange a 
                         -> NeuralNet a 
                         -> (NeuralNet a, WeightChange a)
backPropogate config (xs, ds) (WeightChange ls) nn = (new_net, new_dws)
  where (InducedNN nss o) = induce_net (act_func config) nn xs 
        (output_layer:next_layer:reversed_hidden) = reverse nss
        (dw:reversed_dws) = case ls of 
                            [] -> repeat (Layer (repeat (repeat 0))) 
                            (z:zs) -> reverse ls
        (updated_output_layer, output_weight_changes, output_gradients, _) = back_propogate_output config ds dw next_layer output_layer
        (updated_layers, new_layer_dws, _, _) = back_propogate_hidden config (1:xs) reversed_dws (NN (next_layer:reversed_hidden)) ([updated_output_layer], [output_weight_changes], output_gradients, output_layer)
        new_net = NN updated_layers
        new_dws = WeightChange new_layer_dws


-- believed correct
layer_gradients :: (Num a) => ActivationFunction a -> Layer (Neuron a) -> [a] -> [a]
layer_gradients af l errs = zipWith (*) (map phi' ilfs) errs
  where phi' = derivitive af 
        ilfs = map (local_field) (nodes l)

-- believed correct
weight_change :: (Num a) => TrainingConfig a -> Layer [a] -> Layer a -> Layer a -> Layer [a]
weight_change (TrainingConfig af a b) momentum_layer (Layer layer_gradients) (Layer bwd_ys) = Layer $ map (\g -> map (\y -> a * y * g) bwd_ys) layer_gradients

-- believed correct
layer_weight_deltas :: (Num a) => TrainingConfig a -> [[a]] -> [a] -> [a] -> [[a]]
layer_weight_deltas (TrainingConfig _ a b) dwss_old gradients ys = zipWith (zipWith (+)) momentums descents
  where momentums = map (map (* b)) dwss_old
        descents  = map (\g -> map (g *) ys) (map (* a) gradients)
            
-- correctedness == unknown
hidden_layer_errors :: (Num a) => [a] -> [[a]] -> [a]
hidden_layer_errors pgs fwd_wss = map (sum . (zipWith (*) pgs)) fwd_wss


----------------------------------------------------------------
---------------------- Back Propogation Train ------------------
----------------------------------------------------------------

train :: (Num a) => TrainingConfig a -> [([a], [a])] -> (NeuralNet a, WeightChange a) -> (NeuralNet a, WeightChange a)
train c (t:ts) (nn, dws) = train c ts $ backPropogate c t dws nn 
train c [] result = result

-- TODO: I THINK MY ERROR IS IN THE OUTPUT LAYER COMPUTATION

