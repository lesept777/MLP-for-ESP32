/*
  Test Breats Cancer: predict plant type using attributes
  Information:http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
  Dataset can be found here: http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
  The data was changed as follows
  - remove the first column, supposedly a patient id
  - change the second column: (M,B) --> (1,0) (Malignent, Benign) and move it to the end
  - save as csv file (the file is now in the Data folder)
  The dataset has 30 features per sample, so the learning time may be quite long
  (c) Lesept - September 2020
*/
#define FORMAT_SPIFFS_IF_FAILED true
#include "MLP.h"
const char datasetFile[] = "/BreastCancer.csv";
const char networkFile[] = "/Network_BC.txt";

// Declare the network
int Neurons[] = {30, 20, 10, 1};
int Activations[] = {SIGMOID, SIGMOID, SIGMOID};
MLP Net(4, Neurons, 1);

void setup() {
  Serial.begin(115200);
  Serial.println();
  if (!SPIFFS.begin(FORMAT_SPIFFS_IF_FAILED)) {
    Serial.println("SPIFFS Mount Failed");
    return;
  }

  int nData = 569;
  DATASET dataset;
  int ret = Net.createDataset (&dataset, nData);
  Net.readCsvFromSpiffs(datasetFile, &dataset, 0, 10.0f); // Read dataset
  Net.begin (0.8f);                         // Initialize train & test sets
  Net.initLearn (0.9f, 0.5f, 1.0f, 0.8f);   // Set learning parameters
  Net.setActivation (Activations);
  Net.setMaxError (0.20f);                   // Set the stopping criterion
  bool initialize = true; //!Net.netLoad(networkFile);

  // Training
  long heuristics = H_INIT_OPTIM +
                    H_SELE_WEIGH +
                    H_CHAN_WEIGH +
                    H_CHAN_LRATE +
                    H_CHAN_SGAIN;
  Net.setHeuristics(heuristics);
  Net.setHeurInitialize(initialize); // No need to init a new network if we read it from SPIFFS
  // Display the heuristics parameters
  Net.displayHeuristics();

  unsigned long chrono = millis();
  Net.optimize (&dataset, 10, 50, 15);  // Train baby, train...
  Serial.printf("\nActual duration %u ms\n", millis() - chrono);

  // Evaluation
  float threshold = 0.3f;
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
    if (abs(out - dataset.data[k].Out) < 0.3f) Serial.println("OK");
    else Serial.println("NOK");
  }

  // Display the network
  Net.displayNetwork();

  Net.destroyDataset (&dataset);
}

void loop() {
  // NOPE
}
