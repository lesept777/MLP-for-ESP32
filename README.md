# Multi Layer Perceptron library for ESP32
This library is designed to be used with the Arduino IDE.

It implements both training and inference phase on dataset. Dataset can be created in the sketch or read from a csv file. This library is not intended to work with images, only with arrays of floats.

It can address deep networks, made of one input layer, multiple hidden layers and one output layer. The output layer can have one neuron, for regression, or several neurons for classification. In the case of classification, the softmax function is not coded yet, so remain under 4 or 5 output neurons.

This library was inspired by the code from Karsten Kutza (https://courses.cs.washington.edu/courses/cse599/01wi/admin/Assignments/bpn.html).