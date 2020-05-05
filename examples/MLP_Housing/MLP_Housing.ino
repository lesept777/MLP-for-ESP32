/*
  Test Boston House prices
  Dataset found here: https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data
  The dataset describes 13 numerical properties of houses in
  Boston suburbs and is concerned with modeling the price of
  houses in those suburbs in thousands of dollars.
  (c) Lesept - April 2020
*/
#define FORMAT_SPIFFS_IF_FAILED true
#include "MLP.h"
const char datasetFile[] = "/Housing.csv";

// Declare the network
int Neurons[] = {13, 35, 1};
int Activations[] = {SIGMOID, SIGMOID};
MLP Net(3, Neurons, 1);

void setup() {
  Serial.begin(115200);
  Serial.println();
  if (!SPIFFS.begin(FORMAT_SPIFFS_IF_FAILED)) {
    Serial.println("SPIFFS Mount Failed");
    return;
  }

  int nData = 506;
  DATASET dataset;
  int ret = Net.createDataset (&dataset, nData);
  Net.readCsvFromSpiffs(datasetFile, &dataset, nData, 400.0f); // Read dataset
  Net.begin (0.8f);                         // Initialize train & test sets
  Net.initLearn (0.9f, 0.5f, 1.0f, 0.8f);   // Set learnning parameters
  Net.setActivation (Activations);
  Net.setMaxError (0.2f);                   // Set the stopping criterion

  // Training
  int heuristics = H_INIT_OPTIM +
                   H_CHAN_WEIGH +
                   /* H_MUTA_WEIGH + */
                   /* H_CHAN_BATCH + */
                   H_CHAN_LRATE +
                   H_CHAN_SGAIN;
  Net.setHeuristics(heuristics);
  // Display the heuristics parameters
  Net.displayHeuristics();

  unsigned long chrono = millis();
  Net.optimize (&dataset, 5, 2000, 100);  // Train baby, train...
  Serial.printf("\nActual duration %u ms\n", millis() - chrono);

  // Evaluation
  Net.testNet (&dataset, true);
  Net.evaluateNet (&dataset, 3.0f);     // Display results

  // Prediction
  Serial.println();
  float out[0];
  for (int i = 0; i < 10; i++) {
    int k = random(nData);
    Net.predict(&dataset.data[k].In[0], out);
    Serial.printf ("Validation %d: prediction %f, expected %f --> ",
                   i, out[0], dataset.data[k].Out);
    if (abs(out[0] - dataset.data[k].Out) < 3) Serial.println("OK");
    else Serial.println("NOK");
  }

  // Display the network
  Net.displayNetwork();

  Net.destroyDataset (&dataset);
}

void loop() {
// NOPE
}
