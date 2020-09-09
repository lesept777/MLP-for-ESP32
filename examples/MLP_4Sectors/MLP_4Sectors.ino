/*
  Classify points in 4 sectors
  (c) Lesept - April 2020
*/
#include "MLP.h"
#define FORMAT_SPIFFS_IF_FAILED true
const char networkFile[] = "/SectorNetwork.txt";

int sector (float x, float y) {
  return (x >= 0.5) * 2 + (y >= 0.5);
  /*
     this is equivalent to:
    if (x <  0.5 && y < 0.5)  return 0;
    if (x <  0.5 && y >= 0.5) return 1;
    if (x >= 0.5 && y < 0.5)  return 2;
    if (x >= 0.5 && y >= 0.5) return 3;
  */
}

// Declare the network
int Neurons[] = {2, 8, 1};
int Activations[] = {SIGMOID, SIGMOID};
MLP Net(3, Neurons, 1);

void setup() {
  Serial.begin(115200);
  Serial.println();
  if (!SPIFFS.begin(FORMAT_SPIFFS_IF_FAILED)) {
    Serial.println("SPIFFS Mount Failed");
    return;
  }

  int nData = 500;
  DATASET dataset;
  int ret = Net.createDataset (&dataset, nData);
  // draw random points in [0,1]x[0,1] and
  // set output to 0 - 3 depending on position
  for (int i = 0; i < nData; i++) {
    float x = random(100) / 99.;
    float y = random(100) / 99.;
    dataset.data[i].In[0] = x;
    dataset.data[i].In[1] = y;
    dataset.data[i].Out = sector(x, y);
  }

  Net.begin (0.8f);                         // Initialize train & test sets
  Net.initLearn (0.9f, 0.5f, 1.0f, 0.8f);   // Set learning parameters
  Net.setActivation (Activations);
  Net.setMaxError (0.07f);                  // Set the stopping criterion
  bool initialize = true;//!Net.netLoad(networkFile);

  // Training
  long heuristics = H_INIT_OPTIM +
                    H_CHAN_WEIGH +
                    H_CHAN_BATCH +
                    H_CHAN_LRATE +
                    H_CHAN_SGAIN +
                    H_SELE_WEIGH +
                    H_FORC_S_G_D;
  Net.setHeuristics(heuristics);
  // Test regularization
  // Net.setHeurRegulL2 (true, 3.0);
  // Net.setHeurRegulL1 (true, 100);
  Net.setHeurInitialize(initialize); // No need to init a new network if we read it from SPIFFS
  // Display the heuristics parameters
  Net.displayHeuristics();

  unsigned long chrono = millis();
  Net.optimize (&dataset, 5, 2000, 50);  // Train baby, train...
  Serial.printf("\nActual duration %u ms\n", millis() - chrono);

  // Evaluation
  Net.testNet (&dataset, true);
  Net.evaluateNet (&dataset, 0.4f);     // Display results
  Net.netSave(networkFile);
  Net.destroyDataset (&dataset);

  // Prediction
  Serial.println();
  float out, x[2];
  for (int i = 0; i < 20; i++) {
    x[0] = random(100) / 99.;
    x[1] = random(100) / 99.;
    int expected = sector(x[0], x[1]);
    out = Net.predict(&x[0]);
    int n = (int)(out + 0.5f);
    Serial.printf ("Validation %2d: expected %d, prediction %d -->",
                   i, expected, n);
    if (expected == n) Serial.println("OK");
    else Serial.println("NOK");
  }

  // Display the network
  Net.displayNetwork();
}

void loop() {
  // NOPE
}
