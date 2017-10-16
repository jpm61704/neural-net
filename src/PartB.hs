module PartB where 

import NeuralNetLib
import Neuron
import Utility
import IOTools
import Data.List.Split
import Data.List
import Control.Exception

partBConfig :: TrainingConfig Double
partBConfig = TrainingConfig (logisticActivation 0.2) (0.7) (0.3)

load_partB_data :: IO [([Double], [Double])]
load_partB_data = do 
  contents <- readFile "data/Two_Class_FourDGaussians500.txt"
  return $ parseTrainingDataPB contents

parseTrainingDataPB :: String -> [([Double],[Double])]
parseTrainingDataPB str = zip is ds
  where lns   = map (delete '\r') $ lines str 
        wrdss = map (wordsBy (== ' ')) lns
        parsed_words = map (map read) wrdss
        rawds = map (last) parsed_words 
        ds    = map (\x -> if (x == 1) then [1,0] else [0,1]) rawds
        is    = map init parsed_words
        
partBNet :: IO (NeuralNet Double) 
partBNet = randomWeightsAndBiases [4,4,2]

class1data :: [([Double],[Double])] -> [([Double],[Double])]
class1data xs = assert ((length xs) == 1000) $ (reverse . (drop 500) . reverse) xs

class2data :: [([Double],[Double])] -> [([Double],[Double])]
class2data xs = assert ((length xs) == 1000) $ drop 500 xs 

partB :: IO ()
partB = do 
  net <- partBNet 
  training_data <- load_partB_data 
  let config = TrainingConfig (logisticActivation 0.05) (0.7) (0.3) 
  conduct_training training_data (0.038) config net
  return ()


evaluate_net :: TrainingConfig Double -> NeuralNet Double -> [([Double],[Double])] -> [Bool] -> Double
evaluate_net config net ((xs,ds):t_data) correct_so_far = evaluate_net config net t_data (isCorrect : correct_so_far)
  where os = class_limiter $ output $ induce_net (act_func config) net xs
        isCorrect = os == ds
evaluate_net _ _ [] isCorrects = (fromIntegral numMatches) / (fromIntegral total)
  where numMatches = foldl (\acc p -> case p of { True -> acc + 1 ; False -> acc }) 0 isCorrects
        total      = length isCorrects

partB_cross_validation :: IO ()
partB_cross_validation = do 
  p2_data <- load_partB_data 
  let (training_data, validation_data) = (c1t ++ c2t, c1v ++ c2v)
        where (c1, c2)   = (class1data p2_data, class2data p2_data)
              split_data = splitPlaces [400,100]
              (c1t:c1v:[])  = split_data c1
              (c2t:c2v:[])  = split_data c2
  net <- partBNet
  t_result <- conduct_training training_data (0.038) partBConfig net
  let trained_net = (fst . result) t_result
  let percent_correct = evaluate_net (partBConfig) trained_net validation_data []
  putStrLn (show percent_correct)
  return ()

class_limiter :: [Double] -> [Double]
class_limiter (x:y:[]) = if (x > y) then [1,0] else [0,1]
class_limiter _ = error "output list size > 2"
