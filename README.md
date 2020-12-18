# Multi Layer Perceptron library for ESP32
This library is designed to be used with the Arduino IDE.

Multilayer perceptron is decribed [here](https://en.wikipedia.org/wiki/Multilayer_perceptron). This library implements both training and inference phases on a dataset. Datasets can be created in the sketch or read from a csv file. This library is not intended to work with images, only with arrays of floats.

It can address **deep networks**, made of one input layer, multiple hidden layers and one output layer. The output layer can have one neuron for regression, or several neurons for classification (use SIGMOID or SOFTMAX activation for the last layer).

This library was inspired by the work from [Karsten Kutza](https://courses.cs.washington.edu/courses/cse599/01wi/admin/Assignments/bpn.html). It has changed a lot since...

## Dependencies
* [Arduino FS](https://github.com/espressif/arduino-esp32/tree/master/libraries/FS)
* [ESP32 SPIFFS](https://github.com/espressif/arduino-esp32/tree/master/libraries/SPIFFS)

## Quick start
If you want to test it quickly, try the ["sinus" example](https://github.com/lesept777/MLP-for-ESP32/tree/master/examples/MLP_Sinus)

# Guidelines
## Declare a network
To declare a network, just create an array of int with the number of neurons in each layer. The arguments of the constructor are: number of layers, array of neurons, verbose level.
```
// Declare the network
int Neurons[] = {2, 20, 1}; // Number of neurons in each layer (from input to output)
MLP Net(3, Neurons, 1);     // number of layers, array of neurons, verbose level
```

## Create a dataset
To create a dataset, i.e. allocate memory for the data used for training the network, declare the dataset and call the method `createDataset`. The arguments are the dataset and the number of data (for example the number of lines of the csv file).
```
DATASET dataset;            // Declare the dataset
int nData = 300;            // Number of data
int ret = Net.createDataset (&dataset, nData);
```
The structure of a single data is: an array of floats, the output. The array has as many elements as the number of neurons of the input layer.

The dataset can be filled by reading data from a csv file (stored in SPIFFS) using the method `readCsvFromSpiffs` or by putting values in it. Example for an input layer with 2 neurons:
```
for (int i = 0; i < nData; i++) {
  dataset.data[i].In[0] = ...;
  dataset.data[i].In[1] = ...;
  dataset.data[i].Out   = ...;
}
```

## Initialize the network
Define the parameters of the network.
```
Net.begin (0.8f);
Net.initLearn (0.9f, 0.5f, 1.0f, 0.8f);
int Activations[] = {SIGMOID, SIGMOID};  // See below for available activation functions
Net.setActivation (Activations);
Net.setMaxError (0.3f);                
Net.generateNetwork();
```
* `Net.begin (0.8f)` divides the dataset as 80% training and 20% testing
* `Net.initLearn` defines the values of: the momentum, the learning rate, the gain of the sigmoid activation function, and the change rate of the learning rate.
* `Net.setMaxError` sets the objective output testing error to stop the training phase.

Other options are available (see the examples).

## Run the training phase
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

It is possible to make your own optimization algorithm and train the network by yourself, as shown in the [Sinus2 example](https://github.com/lesept777/MLP-for-ESP32/tree/master/examples/MLP_Sinus2)

## Improve the training
It is possible to improve the training results if the maximum number of epochs is reached. Just save the network, and run the code again (use reset button on ESP32 board):
```
bool initialize = !Net.netLoad(networkFile);
// Training
long heuristics = H_INIT_OPTIM +
                  H_CHAN_WEIGH +
                  H_CHAN_LRATE +
                  H_CHAN_SGAIN;    // See below for available options
Net.setHeuristics(heuristics);
Net.setHeurInitialize(initialize); // No need to init a new network if we read it from SPIFFS
...
Net.netSave(networkFile);
```
On first run, if the save file `networkFile` is not found, the boolean `initialize` is equal to true. The optimization will then begin with random weights. On the next run, the saved network is loaded, and the boolean is set to false. Then, the optimization will begin with the loaded weights, and will try to improve the previous result.

## Inference
When the goal is reached, i.e. when the error made on the test set is lower than the objective, the network is trained. Its parameters can be saved in a file in SPIFFS for later use.

The next phase is inference: run the network on unknown data to predict the output. For the example of a 2 neurons input - 1 neuron output network, it's as simple as:
```
float out, x[2];
x[0] = ...;
x[1] = ...;
out = Net.predict(&x[0]);
```
The float `out` contains the prediction.

## Various options

### **Activation functions** currently available: 
* `SIGMOID`: S-shaped curve, between 0 and 1
* `SIGMOID2`: Similar to `SIGMOID`, but between -1 and +1
* `TANH`: Quite similar to `SIGMOID2`
* `RELU`: Rectified Linear Unit
* `LEAKYRELU` and `ELU` variants
* `SELU` : Scaled Exponential Linear Unit (prevents vanishing & exploding gradient problems)
* `IDENTITY`
* `SOFTMAX`

The **sigmoid** and **hyperbolic tangent** activation functions cannot be used in networks with many layers due to the vanishing gradient problem. In the backpropagation process, gradients tend to get smaller and smaller as we move backwards:  neurons in earlier layers learn slower than neurons in the last layers. This leads to longer learning and less accurate prediction. The **rectified linear** activation function overcomes this problem, allowing models to learn faster and perform better.

![RELU SIGMOID](https://miro.medium.com/max/1452/1*29VH_NiSdoLJ1jUMLrURCA.png "Sigmoid and RELU functions")

**Softmax** for classification problems implemented. `SOFTMAX` can only be used for the last layer. If you choose it, the cost function is [Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy). Otherwise, it's Squared error.

### **Effect of the batch size**
See the ["sinus" example](https://github.com/lesept777/MLP-for-ESP32/tree/master/examples/MLP_Sinus)

### **Optimizor's options**
A set of options is provided to define the heuristics used during training's optimization. They are set by using the `setHeuristics`method or one by one with the associated methods (this enables a finer tuning and the possibility to change the heuristics options during training).

Current options are:
* `H_INIT_OPTIM`   : initialize optimizer with random weights
* `H_CHAN_WEIGH`   : force new random weights during optimization
* `H_SELE_WEIGH`   : select the best of 10 random sets when changing weights
* `H_MUTA_WEIGH`   : slightly change the weights during optimization
* `H_CHAN_BATCH`   : enable to change batch size
* `H_CHAN_LRATE`   : enable to change the learning rate
* `H_CHAN_SGAIN`   : enable to change the sigmoid gain
* `H_CHAN_ALPHA`   : enable to change the momentum (a way to prevent the algorithm from getting stuck in a local minimum)
* `H_SHUF_DATAS`   : shuffle the dataset
* `H_ZERO_WEIGH`   : force low weights to 0
* `H_STOP_TOTER`   : stop optimization if (test + train) Error < threshold (instead of only test) 
* `H_FORC_S_G_D`   : force stochastic gradient descent
* `H_REG1_WEIGH` and  `H_REG2_WEIGH`   : use L1 and/or L2 regularization for lower weight values, see the ["4 sectors" example](https://github.com/lesept777/MLP-for-ESP32/tree/master/examples/MLP_4Sectors)

Heuristics options can be set like this:
```
  long heuristics = H_INIT_OPTIM +
                    H_CHAN_WEIGH +
                    H_CHAN_BATCH +
                    H_CHAN_LRATE +
                    H_CHAN_SGAIN +
                    H_STOP_TOTER;
  Net.setHeuristics(heuristics);
```
or one by one:
```
Net.setHeurInitialize(true);
Net.setHeurChangeEta(true);
Net.setHeurShuffleDataset(false);
```

## Warning
Keep filenames short, under 16 characters total. Otherwise, you may get strange results, due to the fact that the dataset is not read from SPIFFS for example.

For large datasets (such as the Mines example), you may need to increase some parameters, such as:
in `MLP.h`
```
#define MAX_INPUT    70      // Maximum number of neurons in input layer
```
in `MLP.cpp`, `MLP::readCsvFromSpiffs` (around line 159), increase the length of the buffer array
```
  char buffer[500];
```

## (NEW) Residual Neural Network
A new option is available to implement **RNNs**. A Residual Neural Network adds **skip connections** to provide to each layer *l* additional information from layer *l-2*, instead of only layer *l-1*. This is sketched in the following diagram (from [Wikipedia](https://en.wikipedia.org/wiki/Residual_neural_network "Residual Neural Network")).

![RNN](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/ResNets.svg/330px-ResNets.svg.png "Residual Neural Network")

See the ["MLP_4Sectors_ResNet" example](https://github.com/lesept777/MLP-for-ESP32/tree/master/examples/MLP_4Sectors_ResNet)
