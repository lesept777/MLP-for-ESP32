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

int nData = 300;
int Activations[] = {SIGMOID, SIGMOID, SIGMOID};
int bestNErr = nData;
int bestN[2] = {30, 30};

int area (float x, float y) {
  float R = sqrt(x * x + y * y);
  if (R < 0.666) return 0;
  if (R > 1.333) return 2;
  return 1;
}

void fillDataset (DATASET *dataset, int nData) {
  // draw random points inside the circle of radius 2 and
  // set output to 0 - 2 depending on position
  for (int i = 0; i < nData; i++) {
    float rn01 = random(100) / 99.; // random number in [0-1]
    float theta = -3.14 + rn01 * 6.28;
    float R = random(100) / 50.;
    float x = R * cos(theta);
    float y = R * sin(theta);
    dataset->data[i].In[0] = x;
    dataset->data[i].In[1] = y;
    dataset->data[i].Out = area(x, y);
  }
}

bool runNetwork (int n0, int n1) {
  unsigned long chrono = millis();
  if (n0 < 1) n0 = 1;
  if (n1 < 1) n1 = 1;
  int Neurons[] = {2, n0, n1, 1};
  MLP Net(4, Neurons, 0);  // silent
  DATASET dataset;
  Net.createDataset (&dataset, nData);
  fillDataset (&dataset, nData);
  Net.begin (0.8f);                         // Initialize train & test sets
  Net.initLearn (0.9f, 0.5f, 1.0f, 0.8f);   // Set learning parameters
  Net.setActivation (Activations);
  // Training
  long heuristics = H_INIT_OPTIM;
  Net.setHeuristics(heuristics);
  Net.optimize (&dataset, 8, 20, 20);   // Train baby, train...
  Net.evaluateNet (&dataset, 0.4f);     // Display results
  float trainError, testError;
  int nTrainError, nTestError;
  Net.getError (&trainError, &testError, &nTrainError, &nTestError);
  Serial.printf("Dimensions : %2d, %2d --> ", n0, n1);
  Serial.printf("Number of errors : %3d, %3d (%3d) [%u ms]", nTrainError, nTestError,
                nTrainError + nTestError, millis() - chrono);
  //  Serial.printf("MSE Erreurs : %f, %f\n", trainError, testError);
  Net.destroyDataset (&dataset);

  if (nTrainError + nTestError < bestNErr) {
    bestNErr = nTrainError + nTestError;
    bestN[0] = n0;
    bestN[1] = n1;
    Serial.println(" Best case so far...");
    Net.netSave(networkFile);
    return true;
  }
  Serial.println();
  return false;
}

void setup() {
  Serial.begin(115200);
  Serial.println();
  if (!SPIFFS.begin(FORMAT_SPIFFS_IF_FAILED)) {
    Serial.println("SPIFFS Mount Failed");
    return;
  }


  // Optimization parameters
  int boundaries[] = {30, 20};
  int n0, n1;
  unsigned long chrono = millis();

  // Greedy search
  Serial.println("GREEDY SEARCH... INIT");
  n0 = boundaries[0] / 2;
  n1 = boundaries[1] / 2;
  runNetwork (n0, n1);

  int step = 1;
  Serial.println("\nGREEDY SEARCH... NOW SEARCHING");
  for (int i = 0; i < 20; i++) {
    Serial.printf ("Case number %d\n", i + 1);
    n0 = bestN[0];
    n1 = bestN[1];
    bool case1 = runNetwork (n0 - step, n1 - step);
    bool case2 = runNetwork (n0 + step, n1 - step);
    bool case3 = runNetwork (n0 - step, n1 + step);
    bool case4 = runNetwork (n0 + step, n1 + step);
    if (!case1 && !case2 && !case3 && !case4) {
      step *= 2;
      if (step > 8) break;
      Serial.printf("Changing step to %d\n", step);
    }
    Serial.println();
  }

  Serial.printf("Best case : % d, % d, %d errors\n", bestN[0],  bestN[1], bestNErr);
  Serial.println("Training...");

  // Run best case:
  int Neurons[] = {2, bestN[0], bestN[1], 1};
  MLP Net(4, Neurons, 2);  // Little more talkative
  DATASET dataset;
  Net.createDataset (&dataset, nData);
  fillDataset (&dataset, nData);
  Net.begin (0.8f);                         // Initialize train & test sets
  Net.initLearn (0.9f, 0.5f, 1.0f, 0.8f);   // Set learning parameters
  Net.setActivation (Activations);
  // Training
  Net.netLoad(networkFile);
  long heuristics = H_INIT_OPTIM +
                    H_CHAN_WEIGH +
                    H_CHAN_LRATE +
                    H_CHAN_SGAIN;
  Net.setHeuristics(heuristics);
  Net.optimize (&dataset, 10, 400, 50); // Train baby, train...
  Serial.printf("\nActual duration % u ms\n", millis() - chrono);

  Net.evaluateNet (&dataset, 0.4f);     // Display results
  Net.netSave(networkFile);
  Net.destroyDataset (&dataset);

  // Prediction for random points in the square [0-2]x[0-2]
  Serial.println();
  for (int i = 0; i < 20; i++) {
    float x[2];
    x[0] = random(100) / 50.;
    x[1] = random(100) / 50.;
    int expected = area(x[0], x[1]);
    float out = Net.predict(&x[0]);
    int n = (int)(out + 0.5f);
    Serial.printf ("Validation % 2d: expected % d, prediction % d -- > ",
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
