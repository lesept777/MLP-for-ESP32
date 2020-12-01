# Sonar imagery: detection of Mines vs. Rocks

This is the data set used by Gorman and Sejnowski in their study of the classification of sonar signals using a neural network.  The
task is to train a network to discriminate between sonar signals bounced off a metal cylinder and those bounced off a roughly cylindrical rock (see the information.txt file).

It's an example of managing a large dataset : 208 lines of 60 inputs. Some changes were made in the library to handle this big dataset:
in `MLP.h`
```
#define MAX_INPUT    70      // Maximum number of neurons in input layer
```
in `MLP.cpp`, `MLP::readCsvFromSpiffs` (around line 159), increase the length of the buffer array
```
  char buffer[500];
```

A good accuracy is obtained in 20 seconds, with 71 training epochs.


## REFERENCES: 
Gorman, R. P., and Sejnowski, T. J. (1988).  "Analysis of Hidden Units
in a Layered Network Trained to Classify Sonar Targets" in Neural Networks,
Vol. 1, pp. 75-89.