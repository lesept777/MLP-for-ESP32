# Fitting a sine curve, user's optimization
This example shows how to fit the sine curve, for x from -PI to +PI with user's training instructions.

As for the sinus example, the network has 4 layers, with neurons {1, 8, 3, 1}. The dataset creation is exactly the same, as well as the training parameters

## Training 
First initialize the weights with random values. Then shuffle the complete dataset and set the mini-batch size. 
```
  float trainError, testError;
  Net.randomWeights (0.5f);
  Net.shuffleDataset (&dataset, 0, nData);
  Net.setBatchSize (30);
  float minError = 10;
```
A maximum of 5000 epochs are run. One epoch is made of the `trainNet` and `testNet` methods. `trainNet` does the propagation - backpropagation process, and `testNet` computes the errors. If a better set of weights (lower error on the test set) is found, its value is displayed. 
```
  for (int e = 0; e < 5000; e++) {
    Net.trainNet (&dataset);
    Net.testNet (&dataset, true);
```
If the maximum number of epochs is reached or the error is lower than a threshold (0.002), the optimization process is stopped.
```
    Net.getError (&trainError, &testError);
    if (testError < minError) {
      minError = testError;
      Serial.printf("Epoch %4d Error = %.3f\n", e,minError);
    }
    if (testError < 0.002) break;
  }
```

# Evaluation
Evaluation provides the number of errors on the training and testing sets. The threshold for the error is set to 0.05.
```
Net.testNet (&dataset, true);
Net.evaluateNet (&dataset, 0.05f);     // Display results
```
The result is shown in the file [Output.txt](https://github.com/lesept777/MLP-for-ESP32/blob/master/examples/MLP_Sinus2/Output.txt)
