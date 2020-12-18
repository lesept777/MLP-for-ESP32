# Example of ResNet
A new option is available to implement **RNNs**. A Residual Neural Network adds **skip connections** to provide to each layer *l* additional information from layer *l-2*, instead of only layer *l-1*. This is sketched in the following diagram (from [Wikipedia](https://en.wikipedia.org/wiki/Residual_neural_network "Residual Neural Network")).

![RNN](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/ResNets.svg/330px-ResNets.svg.png "Residual Neural Network")

To enable ResNet, just add **true** at the end of the network declaration
```
int Neurons[] = {2, 5, 1};
MLP Net(3, Neurons, 1, true);
```

The network is automatically built with additional links from the input layer to the output layer. The output neuron receives the output of both the 5 hidden neurons **and** the 2 input neurons.

The tests show that this option provides several advantages:
* Quicker learning phase (but this may be also due to a better random set of weights)
* Reduce the number of neurons in the hidden layer(s), while keeping similar results
* Reduce the size of the dataset. Using only 300 data in the dataset instead of 500 still enables a successful learning phase (this also enables to reduce the learning time).

The example also shows the effect of L2 regularization on the weights. Without regularization, the weights are much bigger than when using regularization:

Without:
```
Average weight L1 norm: 9.53933 (lambda = 0.000e+00)
Average weight L2 norm: 112.53433 (lambda = 0.000e+00)
```
With: uncomment this line
```
Net.setHeurRegulL2 (true, 3.0);
```
L2 norm is drastically reduced.
```
Average weight L1 norm: 4.37046 (lambda = 0.000e+00)
Average weight L2 norm: 17.09938 (lambda = 3.000e-06)
```