module Main where

import Lib
import NeuralNetLib
import PartA
import System.IO
import Data.List.Split
import Data.List
import IOTools

main :: IO ()
main = do 
  putStrLn "Part A:"
  training_data <- load_data
  conduct_training training_data
  return ()


conduct_training :: [([Double], [Double])] -> IO ()
conduct_training training_data = do 
  let (nn, dws) = train partAConfig training_data (partANet, WeightChange [])
  print_net nn

load_data :: IO [([Double], [Double])]
load_data = do    
      contents <- readFile "data/cross_data.csv" 
      putStr contents
      return $ parseTrainingDataPA contents
      

parseTrainingDataPA :: String -> [([Double], [Double])]
parseTrainingDataPA str = map (\(xs) -> (init xs, [last xs])) doubles 
 where l_strs = map (delete '\r') $ lines str
       l_elem_strs = map (splitOn ",") l_strs
       doubles = map (parseDoubles) l_elem_strs
             
parseDoubles :: [String] -> [Double]
parseDoubles = map read
      

        
        
        
        
        
        
        