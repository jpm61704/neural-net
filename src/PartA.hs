module PartA where 

import NeuralNetLib
import Neuron
import Utility
import IOTools

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

broken_test = print_net (fst (pa_backPropogate [1,1] [1])) 
    
  
pa_backPropogate xs ds = backPropogate partAConfig (xs, ds) (WeightChange []) partANet

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

net1config :: TrainingConfig Double
net1config = TrainingConfig (logisticActivation 1) (1) (0)

backPropogateN1 = backPropogate net1config ([1], [1]) (WeightChange []) net1

forpropN1 = induce_net (logisticActivation 1) net1 [1]

l1 = case forpropN1 of 
  (InducedNN (x:xs) _) -> x 
  _ -> Layer []
