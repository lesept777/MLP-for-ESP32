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
Training is done by a single line: optimize the network on the `dataset` on 1 iteration, made of 4000 epochs (maximum) and batch size of 40 data.
```
Net.optimize (&dataset, 1, 4000, 40);  // Train baby, train...
```
Evaluation: provides the number of errors on the training and testing sets. The threshold for the erro is set to `0.05`.
```
Net.testNet (&dataset, true);
Net.evaluateNet (&dataset, 0.05f);     // Display results
```
# The effect of batch size
As the convergence is quick, I used this example to see the effect of the batch size on the speed of convergence. The following table shows the mean numbers of epochs and learning time averaged on 10 trials, for various batch sizes. The dataset is made of 300 samples, and training is stopped when the error is below 0.005.
|  Batch size |   Epochs   |  Duration (ms)  |
|:-----------:|:----------:|:---------------:|
|	  1	|	1727	|	24212	|
|	  5	|	1516	|	22431	|
|	 10	|	1219	|	19139	|
|	 15	|	1419	|	23553	|
|	 20	|	1715	|	29795	|
|	 25	|	1203	|	21965	|
|	 50	|	1111	|	25202	|
|	 75	|	1429	|	37483	|
|	100	|	1672	|	52874	|

This tends to show that using small mini-batches up to 10 to 25 samples helps converging faster. This is only a tendency, as the obtained results depend on the initial weights, which are randomly chosen.

Another run gives similar tendencies, with lower values around batch size of 35:
|  Batch size |   Epochs   |  Duration (ms)  |
|:-----------:|:----------:|:---------------:|
|	  1	|	1363	|	19097	|
|	  5	|	1498	|	21257	|
|	 10	|	1363	|	20731	|
|	 15	|	1163	|	19102	|
|	 20	|	1471	|	25603	|
|	 25	|	1272	|	23416	|
|	 30	|	1206	|	23202	|
|	 35	|	 888	|	17964	|
|	 40	|	 980	|	20648	|
|	 45	|	 972	|	21255	|
|	 50	|	1505	|	34485	|
|	 60	|	1239	|	30494	|
|	 75	|	1417	|	38773	|
|	100	|	1647	|	52380	|
