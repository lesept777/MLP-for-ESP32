/*
  Test Mines: classification of sonar signals using a neural network.  The objective is to
  discriminate between sonar signals bounced off a metal cylinder (a mine) and those bounced
  off a roughly cylindrical rock.
  The dataset has 208 lines and 61 columns, the last column is 0 = Rock, 1 = Mine
  (c) Lesept - November 2020
*/
#define FORMAT_SPIFFS_IF_FAILED true
#include "MLP.h"
const char datasetFile[] = "/sonar.csv";
const char networkFile[] = "/Network_Mines.txt";

// Declare the network
int Neurons[] = {60, 10, 2};
int Activations[] = {SIGMOID, SIGMOID};
MLP Net(3, Neurons, 2);

void setup() {
  Serial.begin(115200);
  Serial.println();
  if (!SPIFFS.begin(FORMAT_SPIFFS_IF_FAILED)) {
    Serial.println("SPIFFS Mount Failed");
    return;
  }

  int nData = 208;
  DATASET dataset;
  int ret = Net.createDataset (&dataset, nData);
  Net.readCsvFromSpiffs(datasetFile, &dataset, nData, 1.0f); // Read dataset
  Net.begin (0.8f);                         // Initialize train & test sets
  Net.initLearn (0.9f, 0.5f, 1.0f, 0.8f);   // Set learning parameters
  Net.setActivation (Activations);
  Net.setMaxError (0.2f);                   // Set the stopping criterion
  bool initialize = !Net.netLoad(networkFile);

  // Training
  long heuristics = H_INIT_OPTIM +
                    H_CHAN_WEIGH +
                    H_MUTA_WEIGH +
                    H_CHAN_BATCH +
                    H_CHAN_LRATE +
                    H_CHAN_SGAIN;
  Net.setHeuristics(heuristics);
  Net.setHeurInitialize(initialize); // No need to init a new network if we read it from SPIFFS
  // Display the heuristics parameters
  Net.displayHeuristics();

  unsigned long chrono = millis();
  Net.optimize (&dataset, 10, 50, 1);  // Train baby, train...
  Serial.printf("\nActual duration %u ms\n", millis() - chrono);

  // Evaluation
  float threshold = 0.15f;
  Net.testNet (&dataset, true);
  Net.evaluateNet (&dataset, threshold);     // Display results
  Net.netSave(networkFile);

  // Prediction
  Serial.println();
  for (int i = 0; i < 10; i++) {
    int k = random(nData);
    float out = Net.predict(&dataset.data[k].In[0]);
    Serial.printf ("Validation %d: prediction %f, expected %f --> ",
                   i, out, dataset.data[k].Out);
    if (abs(out - dataset.data[k].Out) < 0.35f) Serial.println("OK");
    else Serial.println("NOK");
  }

  // Display the network
  Net.displayNetwork();

  Net.destroyDataset (&dataset);
}

void loop() {
  // NOPE
}
