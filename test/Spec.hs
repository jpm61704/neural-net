import NeuralNetLib 
import Neuron
import Utility
import Test.Hspec

main :: IO ()
main = hspec $ do 
  describe "Neuron" $ do
    describe "induced_local_field" $ do 
      it "ilf zero-weight neuron is zero" $ do 
        (induced_local_field (Neuron [0,0,0]) [1,2,123]) `shouldBe` 0
      it "ilf should be 5" $ do
        (induced_local_field (Neuron [1,(0.5),2]) [(0.5),5,1]) `shouldBe` 5
  describe "Induced Neuron" $ do 
    describe "induce" $ do 
      it "induced neuron holds true ilf" $ do
        (local_field (induce [(0.5),5,1] (Neuron [1,(0.5),2]))) `shouldBe` 5
  describe "Induced Net" $ do 
    it "output of partANet on i=[1..] is 0.6284"$ do
      output (induce_net (logisticActivation 1) partANet [1,1,1,1,1,1,1,1,1,1,1]) `shouldBe` [0.6283670342181629]
      
        

partANet :: NeuralNet Double
partANet = NN [
    Hidden [
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
    Output [
      Neuron [0.1131,
              0.0511, 0.1611, 0.0238, -0.0267, 0.1089,
              0.2381, 0.0784, 0.003, 0.1646, -0.1779 
             ]
      ]
  ]