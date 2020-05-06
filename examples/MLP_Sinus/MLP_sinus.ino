/*
  Test Sinus with optimize (automatic heuristics)
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
  Net.setMaxError (0.002f);

  // Training
  long heuristics = H_INIT_OPTIM +
                   H_CHAN_WEIGH +
                   /* H_MUTA_WEIGH + */
                   H_CHAN_BATCH +
                   H_CHAN_LRATE +
                   H_CHAN_SGAIN +
                   H_CHAN_ALPHA +
                   H_SHUF_DATAS ;
  Net.setHeuristics(heuristics);
  // Display the heuristics parameters
  Net.displayHeuristics();

  unsigned long chrono = millis();
  Net.optimize (&dataset, 1, 4000, 40);  // Train baby, train...
  Serial.printf("\nActual duration %u ms\n", millis() - chrono);

  // Evaluation
  Net.testNet (&dataset, true);
  Net.evaluateNet (&dataset, 0.05f);     // Display results

  // Prediction
  Serial.println();
  float out[0];
  for (int i = 0; i < 10; i++) {
    int k = random(99);
    float x = -3.14f + k * 2.0f * 3.14f / 98.0f;
    Net.predict(&x, out);
    Serial.printf ("Validation x = % .3f : \tprediction % .3f, \texpected % .3f --> ",
                   x, out[0], sin(x));
    if (abs(out[0] - sin(x)) < 0.05f) Serial.println("OK");
    else Serial.println("NOK");
  }

  // Display the network
  Net.displayNetwork();

  Net.destroyDataset (&dataset);
}

void loop() {

}
