module NeuralNetLib where 

import Neuron
import Utility
import Data.List

data NeuralNet a = NN { 
                      net :: [Layer (Neuron a)] }
                 | InducedNN {
                      net :: [Layer (Neuron a)],
                      output :: [a] } 


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
    
----------------------------------------------------------------
------------------- Forward Propogation ------------------------
----------------------------------------------------------------

induce_net :: (Num a) => ActivationFunction a -> NeuralNet a -> [a] -> (NeuralNet a)
induce_net af nn input = InducedNN in_net o 
  where layers = net nn 
        first_layer_induced = fmap (induce input) (head layers)
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

newtype WeightChange a = WeightChange [[[a]]]

backPropogate :: (Num a) => TrainingConfig a             -- config
                         -> ([a], [a])                   -- training_data@(input, desired outputs)
                         -> WeightChange a               -- last set of weight updates
                         -> NeuralNet a                  -- net to backprop
                         -> (NeuralNet a, WeightChange a)  -- updated net
backPropogate config t@(xs, ds) dw nn = undefined
  where induced_net = induce_net (act_func config) nn xs
  


backPropLayers :: (Num a) => TrainingConfig a                -- config
                          -> [a]                             -- desired output of output layer 
                          -> WeightChange a                  -- past iterations weight changes
                          -> [a]                             -- previous layers gradients
                          -> [[a]]                           -- previous layers weights 
                          -> NeuralNet a                     -- the induced net to evaluate
                          -> (NeuralNet a, WeightChange a)   -- updated net and weight deltas
backPropLayers config ds (WeightChange dw) pg plw nn@(InducedNN layers output) = (result_nn, result_dw)
  where result_nn = NN (remaining_layers ++ [layer_type layer_ns])
        result_dw = WeightChange (remaining_dws ++ [layer_dws])
        (NN remaining_layers, WeightChange remaining_dws) = backPropLayers config ds (WeightChange (init dw)) layer_gs (layer_weights layer) (InducedNN (init layers) output)
        (layer_type, (layer_ns, layer_dws, layer_gs)) = case layer of 
          (Hidden ns) -> (Hidden, hidden_layer_step config (last dw) pg plw ((last . init) layers) layer)
          (Output ns) -> (Output, output_layer_step config (last dw) ds output ((last . init) layers) layer )
        layer = last layers

hidden_layer_step :: (Num a) => TrainingConfig a
                             -> [[a]]                   -- last weight updates (for momentum)
                             -> [a]                     -- forward layer gradients
                             -> [[a]]                   -- forward weights from current
                             -> Layer (Neuron a)        -- backward from current
                             -> Layer (Neuron a)        -- layer to compute
                             -> ([Neuron a], [[a]], [a]) 
hidden_layer_step config dws_old flgs plw (Hidden prev_n) (Hidden ns) = (ns', dws', grads') 
  where ns'      = map (\(n,delta_ws)-> Neuron (zipWith (+) (weights n) (delta_ws))) zipped_ns 
          where zipped_ns = zip ns dws'
        dws'     = map (\(neuron, gradient, dnw_old) -> map (\(w, yi, dwij) -> (b * dwij) + (a * gradient * yi)) (zip3 (weights neuron) prev_ys dnw_old)  ) zipped_ns
          where b         = beta config 
                a         = alpha config
                zipped_ns = zip3 ns grads' dws_old
                prev_ys   = map ((activation (act_func config)) . local_field) prev_n
        grads'   = zipWith (*) phi_ilfs err_term
          where phi_ilfs  = map ((derivitive . act_func) config) ilfs
                ilfs      = map local_field ns
                err_term  = map (sum . (zipWith (*) flgs)) (transpose plw)

layer_weights :: Layer (Neuron a) -> [[a]] 
layer_weights l = map weights (nodes l)

-- input is forward layer
-- output is list of forward weights for each neuron
layer_forward_weights :: (Num a) => Layer (Neuron a) -> [[a]]
layer_forward_weights = transpose . layer_weights


output_layer_step :: (Num a) => TrainingConfig a
                             -> [[a]]                   -- last weight updates (for momentum)
                             -> [a]                     -- desired outputs ds
                             -> [a]                     -- nn output
                             -> Layer (Neuron a)        -- next layer back from current
                             -> Layer (Neuron a)        -- layer to compute
                             -> ([Neuron a], [[a]], [a])  
output_layer_step config dws_old ds os (Hidden prev_ns) (Output ns) = (ns', dws', grads') 
  where ns'      = map (\(n,delta_ws)-> Neuron (zipWith (+) (weights n) (delta_ws))) zipped_ns 
          where zipped_ns = zip ns dws'
        dws'     = map (\(neuron, gradient, dnw_old) -> map (\(w, yi, dwij) -> (b * dwij) + (a * gradient * yi)) (zip3 (weights neuron) prev_ys dnw_old)  ) zipped_ns
          where b         = beta config 
                a         = alpha config
                zipped_ns = zip3 ns grads' dws_old
                prev_ys   = map ((activation (act_func config)) . local_field) prev_ns
        grads'   = zipWith (*) errs phi_ilfs
          where errs = zipWith (-) ds os
                phi_ilfs = map ((derivitive . act_func) config) ilfs
                ilfs     = map local_field ns
                

       
        



-- TODO ::  figure out how to zip or compute the layer's weight deltas


-- try a recursion approach
          
-- I believe that I need to save the previous layer in a tuple during this fold in order to see one layer forward

