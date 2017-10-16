module Main where

import Lib
import NeuralNetLib
import PartA
import System.IO
import Data.List.Split
import Data.List
import IOTools
import Text.Printf
import Utility


main :: IO ()
main = do 
  putStrLn "Part A:"
  --training_data <- load_partA_data
  -- uncomment below for standard part a
      -- conduct_training training_data (0.001) partAConfig partANet
  -- uncomment below for part a number 1, single epoch
      -- print_net $ fst . result $ train_epoch partAConfig training_data (partANet, WeightChange []) 
  -- uncomment below for part a number 3: random initial weights 
  partARandom
  -- partAZero
  return ()

partARandom :: IO ()
partARandom = do 
  net <- randomWeightsAndBiases [2,10,1]
  training_data <- load_partA_data
  config <- setConfig 
  conduct_training training_data (0.001) config net

partAZero :: IO ()
partAZero = do 
  let net = zeroedWeightandBiases [2,10,1]
  training_data <- load_partA_data
  config <- setConfig 
  conduct_training training_data (0.001) config net 

customPartA :: IO ()
customPartA = do
  putStrLn "Part A:"
  training_data <- load_partA_data
  config <- setConfig 
  threshhold <- setThreshhold
  conduct_training training_data threshhold config partANet
  
setConfig :: IO (TrainingConfig Double)
setConfig = do 
  putStrLn "set learning rate(alpha)"
  a <- readLn 
  putStrLn "set momentum rate(beta)"
  b <- readLn
  if b < a 
    then return (TrainingConfig (logisticActivation 1) a b)
    else error "learning rate must be greater than momentum"

setThreshhold :: IO (Double)
setThreshhold = do 
  putStrLn "when should training stop? (Default Avg Error Energy = 0.001)"
  readLn 

conduct_training :: [([Double], [Double])] -> Double -> TrainingConfig Double -> NeuralNet Double -> IO ()
conduct_training training_data cutoff config net = conduct_training' training_data cutoff config (net, WeightChange []) 1

conduct_training' :: [([Double], [Double])] -> Double -> TrainingConfig Double -> (NeuralNet Double, WeightChange Double) -> Integer -> IO ()
conduct_training' training_data cutoff config net_and_momentum i = do 
  let tr = train_epoch config training_data net_and_momentum
  printf ("%.4f\n") (avg_error_energy tr)
  if (avg_error_energy tr) < cutoff
    then print_results tr i
    else do 
      training_data' <- shuffle training_data
      conduct_training' training_data' cutoff config (result tr) (i + 1)



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
      

        
        
        
        
        
        
        