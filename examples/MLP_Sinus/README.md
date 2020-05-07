# Fitting a sine curve
This example shows how to fit the sine curve, for x from -PI to +PI with automatic training.

The network used here has 4 layers, with neurons `{1, 8, 3, 1}`.
First create the dataset and put data inside.
```
  // Dataset creation
  int nData = 300;
  DATASET dataset;
  int ret = Net.createDataset (&dataset, nData);
  for (int i = 0; i < nData; i++) {
    float x = -3.14f + i * 2.0f * 3.14f / (nData - 1.0f);
    dataset.data[i].In[0] = x;
    dataset.data[i].Out = sin(x);
  }
  ```
  Then define the training parameters:
  ```
  Net.begin (0.8f);                         // Initialize train & test sets
  Net.initLearn (0.9f, 0.5f, 1.0f, 0.8f);   // Set learning parameters
  Net.setActivation (Activations);
  Net.setMaxError (0.004f);
```
* 80% of the dataset is for training, 20% for testing
* Learning parameters are:
    * Momentum = 0.9
    * Learning rate = 0.5
    * Sigmoid gain = 1
    * LR change rate = 0.8
* Activations are set to `{SIGMOID, SIGMOID, SIGMOID}`
* Training will stop when the error on the test set is lower than 0.004

Set the heuristics:
```
  long heuristics = H_INIT_OPTIM +
                    H_CHAN_WEIGH +
                    H_CHAN_BATCH +
                    H_CHAN_LRATE +
                    H_CHAN_SGAIN +
                    H_CHAN_ALPHA +
                    H_SHUF_DATAS ;
  Net.setHeuristics(heuristics);
```
* Init with random weights
* Random weights if needed
* Variable batch size
* Variable learning rate
* Variable Sigmoid gain
* Variable momentum
* Shuffle dataset if needed

However, this example is so simple and quick that the heuristics is not used.
# Training and evaluation
Training is done by a single line:
```
Net.optimize (&dataset, 1, 4000, 40);  // Train baby, train...
```
Evaluation: 
```
Net.testNet (&dataset, true);
Net.evaluateNet (&dataset, 0.05f);     // Display results
```
