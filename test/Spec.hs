import NeuralNetLib 
import Neuron
import Utility
import Test.Hspec
import PartA

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
      
        

