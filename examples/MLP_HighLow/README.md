# Classify data above or below a sine curve
This example shows how to classify 2D data compared to a sine curve.

First a dataset is created with random input data chosen in the [0,1]x[0,1] square, and output set to 0 if the point is under the sine curve and 1 if above.

## Training
Training parameters are set and the optimization is run. Results are given in the file [Output.txt](https://github.com/lesept777/MLP-for-ESP32/blob/master/examples/MLP_HighLow/Output.txt). The goal is reached in less than 33 seconds, with 3 iterations.

The results are satisfying, and the net's parameters are stored in a file in the SPIFFS.
```
Net.netSave(networkFile);
```

## Improve the results
If we want to improve the results, i.e. setting a lower goal, just change line 41, for example:
```
Net.setMaxError (0.05f); 
```
And run the sketch again. These 2 lines:
```
  bool initialize = !Net.netLoad(networkFile);
...
  Net.setHeurInitialize(initialize); // No need to init a new network if we read it from SPIFFS
```
make sure that the network is loaded from the file and that the optimization begins with the loaded parameters (i.e. the weights are not randomly set this time).

The improved results are shown in the file [OtherOutput.txt](https://github.com/lesept777/MLP-for-ESP32/blob/master/examples/MLP_HighLow/OtherOutput.txt). After 393 new epochs (3.3 seconds), the testing error is below 0.05

Note: even if the validation set shows NOK results, the classification is correct as predicted values above 0.5 should be considered as 1, and below 0.5 as 0. A prediction accuracy of 0.1 was asked, but 0.5 should be ok:
```
    if (abs(out[0] - expected) < 0.1) Serial.println("OK");
    else Serial.println("NOK");
```