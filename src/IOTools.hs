module IOTools where

import Neuron 
import NeuralNetLib

print_net :: (Show a, Num a) => (NeuralNet a) -> IO ()
print_net nn = do 
    mapM_ (\(n,wss) -> print_layer n wss) zip_wsss
    return ()
  where ls              = net nn 
        wsss            = map layer_weights ls 
        zip_wsss        = zip [1,2..] wsss

        
print_layer :: (Show a) => Int -> [[a]] -> IO ()
print_layer n wss = do 
  putStrLn $ "Layer " ++ (show n)
  mapM_ (\(b:ws) -> putStrLn ("\tNeuron \n\t\tbias = " ++ (show b) ++ "\n\t\tWeights = " ++ (show ws)) ) wss
  return ()