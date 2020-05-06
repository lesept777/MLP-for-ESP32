# Multi Layer Perceptron library for ESP32
This library is designed to be used with the Arduino IDE.

It implements both training and inference phases on a dataset. Datasets can be created in the sketch or read from a csv file. This library is not intended to work with images, only with arrays of floats.

It can address **deep networks**, made of one input layer, multiple hidden layers and one output layer. The output layer can have one neuron for regression, or several neurons for classification. In the case of classification, the softmax function is not coded yet (as of April 2020), so stay under 4 or 5 output neurons.

This library was inspired by the code from Karsten Kutza (https://courses.cs.washington.edu/courses/cse599/01wi/admin/Assignments/bpn.html).

# Dependencies
* Arduino FS https://github.com/espressif/arduino-esp32/tree/master/libraries/FS
* ESP32 SPIFFS https://github.com/espressif/arduino-esp32/tree/master/libraries/SPIFFS

# Declare a network
To declare a network, just create an array of int with the number of neurons in each layer. The arguments of the constructor are: number of layers, array of neurons, verbose level.
```
// Declare the network
int Neurons[] = {2, 20, 1};
MLP Net(3, Neurons, 1);
```

# Create a dataset
To create a dataset, i.e. allocate memory for the data used for training the network, declare the dataset and call the method `createDataset`. The arguments are the dataset and the number of data.
```
DATASET dataset;
int nData = 300;
int ret = Net.createDataset (&dataset, nData);
```
The structure of a single data is: an array of floats, the output. The array has as many elements as the number of neurons of the input layer.

The dataset can be filled by reading data from a csv file (stored in SPIFFS) or by putting values in it. Example for an input layer with 2 neurons:
```
for (int i = 0; i < nData; i++) {
  dataset.data[i].In[0] = ...;
  dataset.data[i].In[1] = ...;
  dataset.data[i].Out   = ...;
}
```

# Initialize the network
Define the parameters of the network.
```
Net.begin (0.8f);
Net.initLearn (0.9f, 0.5f, 1.0f, 0.8f);
int Activations[] = {SIGMOID, SIGMOID};
Net.setActivation (Activations);
Net.setMaxError (0.3f);                
```
* `Net.begin (0.8f)` --> Divide the dataset as 80% training and 20% testing
* `Net.initLearn` defines the values of: the momentum, the learning rate, the gain of the sigmoid activation function, and the change rate of the learning rate.
* `Net.setMaxError` sets the objective output testing error to stop the training phase.

Other options are available (see the examples).

# Run the training phase
The training phase is merely the optimization of the weights to better fit the output of the network and the output of the dataset.

A heuristics is used for optimization, based on the error backpropagation process. Various options can be set for the heuristics.

Then, run the optimization. It can be done automatically:
```
Net.optimize (&dataset, 5, 2000, 50);
```
The parameters are:
* The dataset,
* The number of iterations,
* The number of epochs for each iterations,
* The size of the batch of data.

# Improve the training
It is possible to improve the training results if the maximum number of epochs is reached. Just save the network, and run the code again with heuristics parameter `H_INIT_OPTIM` set to 0:
```
bool initialize = !Net.netLoad(networkFile);
// Training
long heuristics = H_INIT_OPTIM +
                  H_CHAN_WEIGH +
                  H_CHAN_LRATE +
                  H_CHAN_SGAIN;
Net.setHeuristics(heuristics);
Net.setHeurInitialize(initialize); // No need to init a new network if we read it from SPIFFS
...
Net.netSave(networkFile);
```
On first run, if the save file is not found, the boolean `initialize` is true. The optimization will begin with random weights. On the next run, the saved network is loaded, and the boolean is set to false. Then, the optimization will begin with the loaded weights, and will try to improve the previous result.

# Inference
When the goal is reached, i.e. when the error made on the test set is lower than the objective, the network is trained. Its parameters can be saved in a file in SPIFFS for later use.

The next phase is inference: run the network on unknown data to predict the output. For the example of a 2 neurons input - 1 neuron output network, it's as simple as:
```
float out[0], x[2];
x[0] = ...;
x[1] = ...;
Net.predict(&x[0], out);
```
The array `out[0]` contains the prediction.

# Various options
* Activation functions currently available: 
    * `SIGMOID`: S-shaped curve, between 0 and 1
    * `SIGMOID2`: Similar to `SIGMOID`, but between -1 and +1
    * `TANH`: Quite similar to `SIGMOID2`
    * `RELU`: Rectified Linear Unit
    * `IDENTITY`

The **sigmoid** and **hyperbolic tangent** activation functions cannot be used in networks with many layers due to the vanishing gradient problem (they saturate). The **rectified linear** activation function overcomes this problem, allowing models to learn faster and perform better.

RELU: ![RELU function](https://i.imgur.com/gKA4kA9.jpg "RELU function") 
SIGMOID: ![SIGMPOID function](https://i.stack.imgur.com/czEqL.png "Sigmoid function")

**Softmax** to come later, for classification problems.
	