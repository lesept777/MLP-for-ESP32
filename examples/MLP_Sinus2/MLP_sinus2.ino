/*
  Test Sinus with user's heuristics
  (c) Lesept - April 2020
*/
#include "MLP.h"

// Declare the network
int Neurons[4] = {1, 8, 3, 1};
int Activations[3] = {SIGMOID, SIGMOID, SIGMOID};
MLP Net(4, Neurons, 1);  // Number of layers, neurons table, verbose level

void setup() {
  Serial.begin(115200);
  Serial.println();

  // Dataset creation
  int nData = 300;
  DATASET dataset;
  int ret = Net.createDataset (&dataset, nData);
  for (int i = 0; i < nData; i++) {
    float x = -3.14f + i * 2.0f * 3.14f / (nData - 1.0f);
    dataset.data[i].In[0] = x;
    dataset.data[i].Out = sin(x);
  }

  Net.begin (0.8f);                         // Initialize train & test sets
  Net.initLearn (0.9f, 0.5f, 1.0f, 0.8f);   // Set learning parameters
  Net.setActivation (Activations);

  // Training
  unsigned long chrono = millis();
  float trainError, testError;
  Net.randomWeights (0.5f);
  Net.shuffleDataset (&dataset, 0, nData);
  Net.setBatchSize (30);
  float minError = 10;
  for (int e = 0; e < 5000; e++) { // 5000 training epochs
    Net.trainNetSGD (&dataset);
    Net.testNet (&dataset, true);
    Net.getError (&trainError, &testError);
    if (testError < minError) {
      minError = testError;
      Serial.printf("Epoch %4d Error = %.3f\n", e, minError);
    }
    if (testError < 0.002) break;
  }
  Serial.printf("\nTraining duration %u ms\n", millis() - chrono);

  // Evaluation
  Net.testNet (&dataset, true);
  Net.evaluateNet (&dataset, 0.05f);     // Display results

  // Prediction
  Serial.println();
  for (int i = 0; i < 10; i++) {
    int k = random(99);
    float x = -3.14f + k * 2.0f * 3.14f / 98.0f;
    float out = Net.predict(&x);
    Serial.printf ("Validation x = % .3f : \tprediction % .3f, \texpected % .3f --> ",
                   x, out, sin(x));
    if (abs(out - sin(x)) < 0.05f) Serial.println("OK");
    else Serial.println("NOK");
  }

  // Display the network
  Net.displayNetwork();

  Net.destroyDataset (&dataset);
}

void loop() {

}
