module Example where 

import NeuralNetLib
import Neuron
import Utility
import IOTools

net1 :: NeuralNet Double 
net1 = NN [
    Layer [
      Neuron [0,1],
      Neuron [0,2]
    ],
    Layer [
      Neuron [0, 1, 1]
    ]
  ]

n1OutputLayer :: Layer (Neuron Double)
n1OutputLayer = Layer [
  InducedNeuron [0, 1, 1] 1.612 
  ]

n1HiddenLayer :: Layer (Neuron Double)
n1HiddenLayer = Layer [
      InducedNeuron [0,1] 1,
      InducedNeuron [0,2] 2
    ]

n1config :: TrainingConfig Double
n1config = TrainingConfig (logisticActivation 1) (1) (1)

n1BackProp = print_net $ fst (backPropogate n1config ([1], [1]) (WeightChange [Layer [[1.0,1.0],[1.0,1.0]], Layer [[1.0,1.0,1.0]]]) net1)

n1ForProp = induce_net (logisticActivation 1) net1 [1]

n1OutputBackProp = case back_propogate_output n1config [1] (Layer [[],[]]) n1HiddenLayer n1OutputLayer of 
  (layer, _, _, _) -> show layer
  
n1HiddenBackProp = case back_propogate_hidden 
                          n1config 
                          [1.0,1.0] 
                          [Layer [[0.0,0.0],[0.0,0.0]]] 
                          (NN [n1HiddenLayer]) 
                          ([Layer [Neuron [0.023, 1.017, 1.020]]], [Layer [[0.023,0.017,0.020]]], (Layer [0.023]), n1OutputLayer) of 
                  (nn, dws, grads, layer) -> do 
                                      putStrLn $ "Layers: \t" ++ (show nn) ++ "\n\n"
                                      putStrLn $ "delta_ws: \t" ++ (show dws) ++ "\n\n"
                                      putStrLn $ "Gradients: \t" ++ (show grads) ++ "\n\n"
                                      putStrLn $ "layer at start: \t" ++ (show layer) ++ "\n\n"