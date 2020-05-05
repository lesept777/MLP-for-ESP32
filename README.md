# Multi Layer Perceptron library for ESP32
This library is designed to be used with the Arduino IDE.

It implements both training and inference phase on dataset. Dataset can be created in the sketch or read from a csv file. This library is not intended to work with images, only with arrays of floats.

It can address deep networks, made of one input layer, multiple hidden layers and one output layer. The output layer can have one neuron for regression, or several neurons for classification. In the case of classification, the softmax function is not coded yet (as of April 2020), so remain under 4 or 5 output neurons.

This library was inspired by the code from Karsten Kutza (https://courses.cs.washington.edu/courses/cse599/01wi/admin/Assignments/bpn.html).

# Dependencies
Arduino FS https://github.com/espressif/arduino-esp32/tree/master/libraries/FS

ESP32 SPIFFS https://github.com/espressif/arduino-esp32/tree/master/libraries/SPIFFS

# Declare a network
To declare a network, just create an array of int with the number of neurons in each layer. The arguments of the constructor are: number of layers, array of neurons, verbose level.
```
// Declare the network
int Neurons[] = {2, 20, 1};
MLP Net(3, Neurons, 1);
```

# Create a dataset
To create a dataset, i.e. allocate memory for the data used for training the network, declare the dataset and call the method createDataset. The arguments are the dataset and the number of data.
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