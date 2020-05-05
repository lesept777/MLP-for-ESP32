/*
  Classify points inside 3 circles
  Area 0: radius < 0.666
  Area 1 : 0.666 < radius < 1.333
  Area 2 : radius > 1.333
  (c) Lesept - April 2020
*/
#include "MLP.h"
#define FORMAT_SPIFFS_IF_FAILED true
const char networkFile[] = "/CirclesNetwork.txt";

int area (float x, float y) {
  float R = sqrt(x * x + y * y);
  if (R < 0.666) return 0;
  if (R > 1.333) return 2;
  return 1;
}

// Declare the network
int Neurons[] = {2, 20, 1};
int Activations[] = {SIGMOID, SIGMOID};
MLP Net(3, Neurons, 1);

void setup() {
  Serial.begin(115200);
  Serial.println();
  if (!SPIFFS.begin(FORMAT_SPIFFS_IF_FAILED)) {
    Serial.println("SPIFFS Mount Failed");
    return;
  }

  int nData = 300;
  DATASET dataset;
  int ret = Net.createDataset (&dataset, nData);
  // draw random points inside the circle of radius 2 and
  // set output to 0 - 2 depending on position
  for (int i = 0; i < nData; i++) {
    float rn01 = random(100) / 99.; // random number in [0-1]
    float theta = -3.14 + rn01 * 6.28;
    float R = random(100) / 50.;
    float x = R * cos(theta);
    float y = R * sin(theta);
    dataset.data[i].In[0] = x;
    dataset.data[i].In[1] = y;
    dataset.data[i].Out = area(x, y);
  }

  Net.begin (0.8f);                         // Initialize train & test sets
  Net.initLearn (0.9f, 0.5f, 1.0f, 0.8f);   // Set learning parameters
  Net.setActivation (Activations);
  Net.setMaxError (0.3f);                     // Set the stopping criterion
  bool initialize = !Net.netLoad(networkFile);

  // Training
  int heuristics = H_INIT_OPTIM +
                   H_CHAN_WEIGH +
                   /* H_MUTA_WEIGH + */
                   /* H_CHAN_BATCH + */
                   H_CHAN_LRATE +
                   H_CHAN_SGAIN;
  Net.setHeuristics(heuristics);
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

  // Prediction for random points in the square [0-2]x[0-2]
  Serial.println();
  float out[0], x[2];
  for (int i = 0; i < 20; i++) {
    x[0] = random(100) / 50.;
    x[1] = random(100) / 50.;
    int expected = area(x[0], x[1]);
    Net.predict(&x[0], out);
    int n = (int)(out[0] + 0.5f);
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
