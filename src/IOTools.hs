module IOTools where

import Neuron 
import NeuralNetLib
import System.Random
import Data.Array.IO
import Control.Monad
import Text.Printf

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
  
print_results :: TrainingResult -> Integer -> IO ()
print_results (TR avg_error_energy (nn, _)) i = do
  putStrLn (replicate 5 '\n')
  putStrLn "====== Trained Net ======"
  putStr $ concat ["- Took ", (show i), " Iterations\n- Trained to Avg Error Energy "]
  printf "%.8f\n" avg_error_energy
  putStrLn "======   Network   ======"
  print_net nn 
  return ()

zeroedWeightandBiases :: [Int] -> NeuralNet Double 
zeroedWeightandBiases (l1:l2:ls) = NN (layer : layers)
  where layer = Layer $ replicate l2 (Neuron (replicate (l1 + 1) 0.0))
        NN layers = zeroedWeightandBiases (l2:ls)
zeroedWeightandBiases _ = NN []

randomWeightsAndBiases :: [Int] -> IO (NeuralNet Double)
randomWeightsAndBiases (l1:l2:ls) = do 
  layer <- initialize_layer l1 l2
  NN layers <- randomWeightsAndBiases (l2:ls)
  return $ NN (layer : layers)
  where 
        initialize_layer :: Int -> Int -> IO (Layer (Neuron Double))
        initialize_layer num_inputs num_outputs = do 
          neurons <- replicateM num_outputs (random_neuron num_inputs)
          return $ Layer neurons
randomWeightsAndBiases _ = return $ NN []
          
random_neuron :: Int -> IO (Neuron Double)
random_neuron num_weights = do 
  ws <- replicateM (num_weights + 1) randomIO
  return $ Neuron ws 


shuffle :: [a] -> IO [a]
shuffle xs = do
        ar <- newArray n xs
        forM [1..n] $ \i -> do
            j <- randomRIO (i,n)
            vi <- readArray ar i
            vj <- readArray ar j
            writeArray ar j vi
            return vj
  where
    n = length xs
    newArray :: Int -> [a] -> IO (IOArray Int a)
    newArray n xs =  newListArray (1,n) xs