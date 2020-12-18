# Example of ResNet
A new option is available to implement **RNNs**. A Residual Neural Network adds **skip connections** to provide to each layer *l* additional information from layer *l-2*, instead of only layer *l-1*. This is sketched in the following diagram (from [Wikipedia](https://en.wikipedia.org/wiki/Residual_neural_network "Residual Neural Network")).

![RNN](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/ResNets.svg/330px-ResNets.svg.png "Residual Neural Network")

To enable ResNet, just add **true** at the end of the network declaration
```
int Neurons[] = {2, 5, 1};
MLP Net(3, Neurons, 1, true);
```

The network is automatically built with additional links from the input layer to the output layer. The output neuron receives the output of both the 5 hidden neurons and the 2 input neurons.

The tests show that this option provides several advantages:
* Quicker learning phase (but this may be also due to a better random set of weights)
* Reduce the number of neurons in the hidden layer(s), while keeping similar results
* Reduce the size of the dataset. Using only 300 data in the dataset instead of 500 still enables a successful learning phase (this also enables to reduce the learning time).