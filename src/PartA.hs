module PartA where 

import NeuralNetLib
import Neuron
import Utility
import IOTools
import Data.List.Split
import Data.List

partAConfig :: TrainingConfig Double
partAConfig = TrainingConfig (logisticActivation 1) (0.7) (0.3)

partANet :: NeuralNet Double
partANet = NN [
    Layer [
      Neuron [-0.122, 0.4033, -1.0562],
      Neuron [0.9401, 0.39, 0.6544],
      Neuron [0.4271, 0.6376, -0.0601],
      Neuron [-0.1775, 0.0064, -0.0462],
      Neuron [-0.7019, 0.0782, 0.2728],
      Neuron [-0.3326, -0.2115, 1.0252],
      Neuron [-0.6961, 0.7298, -0.5047],
      Neuron [-0.9316, -0.7109, 0.349],
      Neuron [-0.3681, -0.9315, 0.9867],
      Neuron [1.0695, 0.8441, 0.4276]
      ],
    Layer [
      Neuron [0.1131,
              0.0511, 0.1611, 0.0238, -0.0267, 0.1089,
              0.2381, 0.0784, 0.003, 0.1646, -0.1779 
             ]
      ]
  ]

load_partA_data :: IO [([Double], [Double])]
load_partA_data = do    
      contents <- readFile "data/cross_data.csv" 
      return $ parseTrainingDataPA contents
      

parseTrainingDataPA :: String -> [([Double], [Double])]
parseTrainingDataPA str = map (\(xs) -> (init xs, [last xs])) doubles 
 where l_strs = map (delete '\r') $ lines str
       l_elem_strs = map (splitOn ",") l_strs
       doubles = map (parseDoubles) l_elem_strs
       
       parseDoubles :: [String] -> [Double]
       parseDoubles = map read

partARandom :: IO ()
partARandom = do 
  net <- randomWeightsAndBiases [2,10,1]
  training_data <- load_partA_data
  config <- setConfig 
  conduct_training training_data (0.001) config net
  return ()

partAZero :: IO ()
partAZero = do 
  let net = zeroedWeightandBiases [2,10,1]
  training_data <- load_partA_data
  config <- setConfig 
  conduct_training training_data (0.001) config net
  return ()

customPartA :: IO ()
customPartA = do
  putStrLn "Part A:"
  training_data <- load_partA_data
  config <- setConfig 
  threshhold <- setThreshhold
  conduct_training training_data threshhold config partANet
  return ()

