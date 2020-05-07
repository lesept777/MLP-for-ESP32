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
  
