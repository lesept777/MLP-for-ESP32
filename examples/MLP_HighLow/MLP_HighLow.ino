/*
  Classify data above or below a sinus curve
  (c) Lesept - April 2020
*/
#include "MLP.h"
#define FORMAT_SPIFFS_IF_FAILED true
const char networkFile[] = "/Network_HiLo.txt";

#define f(x) (0.5 + 0.3 * sin(4 * 3.14 * x));

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

  int nData = 200;
  DATASET dataset;
  int ret = Net.createDataset (&dataset, nData);
  // draw random points in [0,1]x[0,1] and
  // set output to 0 if under the curve or 1 if over
  for (int i = 0; i < nData; i++) {
    float x = (float)i / (nData - 1.);
    float T = f(x);
    float y = random(100) / 99.;
    dataset.data[i].In[0] = x;
    dataset.data[i].In[1] = y;
    dataset.data[i].Out = (y > T) ? 1 : 0;
  }

  Net.begin (0.8f);                         // Initialize train & test sets
  Net.initLearn (0.9f, 0.5f, 1.0f, 0.8f);   // Set learning parameters
  Net.setActivation (Activations);
  Net.setMaxError (0.1f);                     // Set the stopping criterion
  bool initialize = !Net.netLoad(networkFile);

  // Training
  int heuristics = H_INIT_OPTIM +
                   H_CHAN_WEIGH +
                   /*       H_MUTA_WEIGH +   */
                   H_CHAN_BATCH +
                   H_CHAN_LRATE +
                   H_CHAN_SGAIN;
  Net.setHeuristics(heuristics);
  Net.setHeurInitialize(initialize); // No need to init a new network if we read it from SPIFFS
  // Display the heuristics parameters
  Net.displayHeuristics();

  unsigned long chrono = millis();
  Net.optimize (&dataset, 10, 1000, 50);  // Train baby, train...
  Serial.printf("\nActual duration %u ms\n", millis() - chrono);

  // Evaluation
  Net.testNet (&dataset, true);
  Net.evaluateNet (&dataset, 0.1f);     // Display results
  Net.netSave(networkFile);
  Net.destroyDataset (&dataset);

  // Prediction
  Serial.println();
  float out[1], x[2];
  for (int i = 0; i < 10; i++) {
    x[0] = random(100) / 99.;
    x[1] = random(100) / 99.;
    float T = f(x[0]);
    float expected = (x[1] > T) ? 1 : 0;
    Net.predict(&x[0], out);
    Serial.printf ("Validation %d: prediction %f, expected %f --> ",
                   i, out[0], expected);
    if (abs(out[0] - expected) < 0.1) Serial.println("OK");
    else Serial.println("NOK");
  }

  // Display the network
  Net.displayNetwork();
}

void loop() {
  // NOPE
}
