# Deep-Computer-Go

A Deep Learning Project of the <a href="https://www.lamsade.dauphine.fr/wp/iasd/en/">master IASD</a>. The goal is to train a multi-head network for playing the game of Go. 
In order to be fair about training ressources the number 
of parameters for the networks must be lower than 1 000 000. <a href="https://www.lamsade.dauphine.fr/~cazenave/DeepLearningProject.html">More info.</a>

Top-3 model on intra-master tournament. The architecture is based on attention-like modules such as CBAM and Squeeze-and-Excitation modules as well as 
residual blocks and activation functions for deep networks. 


# Data

The data used for training comes from 
the Katago Go program self played games. There are 1 000 000 different games in total in the training set. The input data is composed of 21 19x19 planes 
(color to play, ladders, current state on two planes, two previous states on four planes). The output targets are the policy (a vector of size 361 with 1.0 
for the move played, 0.0 for the other moves), and the value (1.0 if White won, 0.0 if Black won). 


# Authors

* Emilie Chhean
* Yves Tran
