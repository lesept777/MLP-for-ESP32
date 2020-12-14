# Classify points inside 3 circles
This example shows how to classify 2D data in 3 concentric circles. It's quite a complex problem (see the other example MLP_Circles).

The assumption is made that the network has 4 layers, with 2 hidden layers. But we don't know how many neurons would be best for the hidden layers.

## Greedy searching the numbers of neurons
This example shows how to use a greedy algorithm to optimize the number of neurons in the hidden layers. The best result of the greedy search is for 14 and 11 neurons. After finding these numbers, a new network is built and the training is run.

The total run time is close to 20 minutes, and the validation phase shows no error.