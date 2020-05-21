#include <Arduino.h>
#include "MLP.h"

#define MIN_float      -HUGE_VAL
#define MAX_float      +HUGE_VAL
#define SWAP(x, y)    { typeof((x)) SWAPVal = (x); (x) = (y); (y) = SWAPVal; }

#define BIAS          1
//
/*
  Constructor, arguments are:
    number of layers
    array of number of neurons
    verbose level (0= silent, 1= intermediary, 2=very talkative)
*/
MLP::MLP(int numLayers, int units[], int verbose)
{
  _numLayers = numLayers;
  for (int i = 0; i < _numLayers; i++) _units[i] = units[i];
  _verbose = verbose;
  generateNetwork();
}

MLP::~MLP()
{
  for (int l = 0; l < _numLayers; l++) {
    delete &Layer[l]->Units;
    delete Layer[l]->Output;
    delete Layer[l]->Error;
    // if (l != 0) {
    //   for (int i = 1; i <= _units[l]; i++) {
    //     delete Layer[l]->Weight[i];
    //     delete Layer[l]->WeightSave[i];
    //     delete Layer[l]->dWeight[i];
    //   }
    delete Layer[l]->Weight;
    delete Layer[l]->WeightSave;
    delete Layer[l]->dWeight;
    delete &Layer[l]->Output[0];
    // }
    delete Layer[l];
  }
  delete InputLayer;
  delete OutputLayer;
  delete &Alpha;
  delete &Eta;
  delete &Gain;
  delete &Error;
  delete &EtaSave;
  delete &GainSave;
  delete &AlphaSave;
  delete Layer;
}

// Loads a network from SPIFFS file system
bool MLP::netLoad(const char* const path)
{
  File file = SPIFFS.open(path);
  if (!file || file.isDirectory()) {
    Serial.printf("%s - failed to open file for reading\n", path);
    return 0;
  }
  // Load header.
  if (_verbose > 0) Serial.print("Loading network ");
  _numLayers = readIntFile (file);
  if (_verbose > 1) Serial.printf("-> %d layers \n", _numLayers);
  for (int l = 0; l < _numLayers; l++) {
    _units[l] = readIntFile (file);
    _activations[l] = readIntFile (file);
    if (_verbose > 1) Serial.printf("layer %d -> %d neurons, %s\n",
     l, _units[l], ActivNames[_activations[l]]);
  }
  generateNetwork();
  // Read parameters
  Alpha = readFloatFile (file);
  AlphaSave = Alpha;
  if (_verbose > 1) Serial.printf("Alpha = %f\n", Alpha, 6);
  Eta = readFloatFile (file);
  EtaSave = Eta;
  if (_verbose > 1) Serial.printf("Eta = %f\n", Eta, 6);
  Gain = readFloatFile (file);
  GainSave = Gain;
  if (_verbose > 1) Serial.printf("Gain = %f\n", Gain, 6);

  // Load  weights
  if (_verbose > 1) Serial.print ("Reading weights : ");
  int iW = 0;
  for (int l = 1; l < _numLayers; l++) {
    Layer[l]->Number = l;
    for (int i = 1; i <= Layer[l]->Units; i++) {
      for (int j = 0; j <= Layer[l - 1]->Units; j++) {
        float x = readFloatFile (file);
        Layer[l]->Weight[i][j] = x;
        if (_verbose > 2) Serial.printf("\nLayer %d weigth %d, %d = %f",l,i,j, Layer[l]->Weight[i][j]);
        ++iW;
      }
    }
  }
  if (_verbose > 2) Serial.println();
  if (_verbose > 1) Serial.printf("found %d weights\n", iW);
  file.close();
  Serial.println();
  return 1;
}

// Saves a network to SPIFFS file system
void MLP::netSave(const char* const path)
{
  File file = SPIFFS.open(path, FILE_WRITE);
  if (!file) {
    Serial.printf("%s - failed to open file for writing\n", path);
    return;
  }
  // Save number of layers
  if (_verbose > 0) Serial.printf("Saving network in file %s\n", path);
  file.printf("%d\n", _numLayers);
  if (_verbose > 1) Serial.printf ("%d layers\n", _numLayers);
  for (int l = 0; l < _numLayers; l++) {
    file.printf("%d\n", Layer[l]->Units);
    file.printf("%d\n", Layer[l]->Activation);
    if (_verbose > 1) Serial.printf("Layer %d -> %d neurons, %s\n", l, 
      Layer[l]->Units, ActivNames[Layer[l]->Activation]);
  }
  // Save parameters
  file.printf("%f\n%f\n%f\n", AlphaSave, EtaSave, GainSave);
  if (_verbose > 1) Serial.printf("Alpha = %f\n", AlphaSave);
  if (_verbose > 1) Serial.printf("Eta = %f\n"  , EtaSave);
  if (_verbose > 1) Serial.printf("Gain = %f\n" , GainSave);
  // Save layers
  int iW = 0;
  for (int l = 1; l < _numLayers; l++) {
    for (int i = 1; i <= Layer[l]->Units; i++) {
      for (int j = 0; j <= Layer[l - 1]->Units; j++) {
        file.printf("%.6f\n", Layer[l]->Weight[i][j]);
        ++iW;
      }
    }
  }
  if (_verbose > 1) Serial.printf ("Saved %d weights\n", iW);
  file.close();
}

// Read a dataset from a csv file on SPIFFS
int MLP::readCsvFromSpiffs (const char* const path, DATASET * dataset, 
  int nData, float coeff = 1.0f)
{
  File file = SPIFFS.open(path);
  if (!file || file.isDirectory()) {
    Serial.printf("%s - failed to open file for reading\n", path);
    return 0;
  }
  char buffer[200];
  char * pch;

  // If nData is provided
  if (nData != 0) {
    _nData = nData;
    for (int i = 0; i < nData; i++) {
      if (_verbose == 3) Serial.printf("line %d\n", i);
      // Read a line of the csv file
      int k = 0;
      if (!file.available()) return -1; // Error: unexpected end of file
      char c = file.read();
      while (c != 10) { // CR
        buffer[k++] = c;
        c = file.read();
      }
      buffer[k] = NULL;
      if (_verbose == 3) Serial.printf("%s\n", buffer);
      pch = strtok (buffer, ",;");
      for (int j = 0; j <= _units[0]; j++) { // Read each value in the line
        float x = atof(pch);
        if (_verbose == 3) Serial.printf("%f ", x);
        if (j < _units[0]) dataset->data[i].In[j] = x / coeff;
        else dataset->data[i].Out = x;
        pch = strtok (NULL, ",;");
      }
      if (_verbose == 3) Serial.println();
    }
  } else { // Else scan the file
    // First line : number of cols = 1 + number of input neurons
    _nData = 1;
    int i = 0;
    while (file.available()) { // Read first line
      char c = file.read();
      buffer[i++] = c;
      if (c == 10) break; // CR
    }
    buffer[i] = NULL;
    if (_verbose == 3) Serial.println(buffer);
    int nCols = 0;
    pch = strtok (buffer, ",;");
    while (pch != NULL) { // Count columns
      float x = atof(pch);
      if (i < _units[0]) dataset->data[0].In[nCols] = x / coeff;
      else dataset->data[0].Out = x;
      ++nCols;
      pch = strtok (NULL, ",;");
    }
    if (nCols != _units[0]+1) {
      Serial.println("Problem reading file");
      Serial.printf("Read %d columns, and %d inout neurons require %d columns\n", 
        nCols, _units[0], _units[0]+1 );
      while(1);
    }
    _units[0] = nCols - 1;
    // Next lines
    while (file.available()) {
      i = 0;
      char c = file.read();
      while (c != 10) { // CR
        buffer[i++] = c;
        c = file.read();
      }
      buffer[i] = NULL;
      pch = strtok (buffer, ",;");
      for (int i = 0; i < nCols; i++) {
        float x = atof(pch);
        if (i < nCols - 1) dataset->data[_nData].In[i] = x / coeff;
        else dataset->data[_nData].Out = x;
        pch = strtok (NULL, ",;");
      }
      ++_nData;
    }
  }
  // dataset->nData = _nData;
  // dataset->nInput = _units[0];
  processDataset (dataset);
  if (_verbose > 0) Serial.printf("Read %d data of %d input\n", _nData, _units[0]);
  return _nData;
}

// Allocate the memory for the user's dataset
int MLP::createDataset (DATASET* dataset, int nData)
{
  float *pos = NULL;  // position within a float array
  _nData = nData;

  // creating data vector
  dataset->data = (Data *)calloc(sizeof(Data), nData);
  if (dataset->data == NULL) return -1;

  dataset->nInput = _units[0];
  dataset->nData = nData;

  // creating input vector
  float *pData = (float *)calloc(sizeof(float), _units[0] * nData);
  if (pData == NULL) return -2;

  // Positioning each input on its sub-array
  pos = pData;
  for (int i = 0; i < nData; i++) {
    dataset->data[i].In = pos;
    pos += _units[0];
  }
  return 0;
}

void  MLP::destroyDataset(DATASET* dataset)
{
  free(dataset->data[0].In);
  free(dataset->data);
}

// Display the parameters of the network
void MLP::displayNetwork()
{
  Serial.println("---------------------------");
  Serial.printf ("Network has %d layers:\n", _numLayers);
  int numWeights = 0;
  for (int l = 0; l < _numLayers; l++) {
    Serial.printf("Layer number %d: %d neurons", l, Layer[l]->Units);
    if (l == 0) Serial.println();
    else {
      Serial.printf(", activation %s, ", ActivNames[Layer[l]->Activation]);
      int numW = Layer[l]->Units * (1 + Layer[l - 1]->Units);
      numWeights += numW;
      Serial.printf("%d weights\n", numW);
      if (_verbose > 2 && l != 0) {
        for (int i = 1; i <= Layer[l]->Units; i++)
          for (int j = 0; j <= Layer[l - 1]->Units; j++)
            Serial.printf("\t%d - %d: %f\n",i,j,Layer[l]->Weight[i][j]);
      }
    }
  }
  Serial.printf ("Total number of weights: %d\n", numWeights);
  Serial.printf("Learning rate is: %.3f\n", Eta);
  Serial.printf("Gain is: %.3f\n", Gain);
  Serial.printf("Momentum is: %.3f\n", Alpha);
  Serial.println("---------------------------");
}

// defines dataset parameters
void MLP::begin(float ratio)
{
  _ratio  = ratio;              // ratio of training data vs. testing
  _nTrain = _ratio * _nData;    // size of training dataset
  _nTest  = _nData - _nTrain;   // size of testing dataset
  _batchSize = _nTrain / 20;
  _outDelta = 0.0f;
  _outMinVal = +HUGE_VAL;
  _datasetProcessed = false;
  _eval = false;
}

// initialize learning and optimizing parameters
void MLP::initLearn(float alpha, float eta, float gain, float anneal)
{
  Alpha = alpha;
  AlphaSave = alpha;
  Eta = eta;
  EtaSave = eta;
  Gain = gain;
  GainSave = gain;
  _anneal = anneal;
}

// initialize test parameters
void MLP::setMaxError(float maxErr) {
  _maxErr = maxErr;
}

void MLP::setVerbose (int verbose) {
  /*
     verbose levels:
     0: silent
     1: show progression
     2: details of all training steps
     3: 2 + content of dataset csv file
  */
  _verbose = verbose;
}
void MLP::setIterations (int iters) {
  _iters = iters;
}
void MLP::setEpochs (int epochs) {
  _epochs = epochs;
}
void MLP::setBatchSize (int batchSize) {
  _batchSize = batchSize;
}
void MLP::setAlpha (float alpha) {
  Alpha = alpha;
  AlphaSave = alpha;
}
void MLP::setEta (float eta) {
  Eta = eta;
  EtaSave = eta;
}
void MLP::setGain (float gain) {
  Gain = gain;
  GainSave = gain;
}
void MLP::setAnneal (float anneal) {
  _anneal = anneal;
}
void MLP::setActivation (int activations[]) {
  _activations[0] = 99;
  for (int i = 0; i < _numLayers - 1; i++) _activations[i+1] = activations[i];
}

void MLP::setHeuristics (long heuristics) { 
  _heuristics = heuristics;
  if (_heuristics != 0) {
    _initialize     = _heuristics & H_INIT_OPTIM; // 0x000001
    _changeWeights  = _heuristics & H_CHAN_WEIGH; // 0x000002
    _mutateWeights  = _heuristics & H_MUTA_WEIGH; // 0x000004
    _changeBatch    = _heuristics & H_CHAN_BATCH; // 0x000008
    _changeEta      = _heuristics & H_CHAN_LRATE; // 0x000010
    _changeGain     = _heuristics & H_CHAN_SGAIN; // 0x000020
    _changeAlpha    = _heuristics & H_CHAN_ALPHA; // 0x000040
    _shuffleDataset = _heuristics & H_SHUF_DATAS; // 0x000080
    _zeroWeights    = _heuristics & H_ZERO_WEIGH; // 0x000100
    _stopTotalError = _heuristics & H_STOP_TOTER; // 0x000200
    _selectWeights  = _heuristics & H_SELE_WEIGH; // 0x000400
  }
}

void MLP::displayHeuristics () {
  Serial.println("---------------------------");
  Serial.println("Heuristics parameters:");
  if(_initialize)      Serial.println ("Init with random weights");
  if(_changeWeights)   Serial.println ("Random weights if needed");
  if(_mutateWeights)   Serial.println ("Slighlty change weights if needed");
  if(_changeBatch)     Serial.println ("Variable batch size");
  if(_changeEta)       Serial.println ("Variable learning rate");
  if(_changeGain)      Serial.println ("Variable Sigmoid gain");
  if(_changeAlpha)     Serial.println ("Variable momentum");
  if(_shuffleDataset)  Serial.println ("Shuffle dataset if needed");
  if(_zeroWeights)     Serial.printf  ("Force weights less than %f to zero\n", _zeroThreshold);
  if(_stopTotalError)  Serial.println ("Stop optimization if train + test error under threshold");
  if(_selectWeights)   Serial.println ("Select best weights at init");
  Serial.println("---------------------------");
}

void MLP::setHeurInitialize (bool val) { _initialize = val; }
void MLP::setHeurChangeWeights (bool val, float range = 1.0f) { 
  _changeWeights = val;
  _range         = range;
}
void MLP::setHeurMutateWeights (bool val, float proba = 0.05f, float percent = 0.15f) { 
  _mutateWeights = val;
  _proba         = proba;
  _percent       = percent;
}
void MLP::setHeurChangeBatch (bool val) { _changeBatch = val; }
void MLP::setHeurChangeEta (bool val, float minEta = 0.35f, float maxEta = 1.1f) { 
  _changeEta     = val;
  _minEta        = minEta;
  _maxEta        = maxEta;
}
void MLP::setHeurChangeAlpha (bool val, float minAlpha = 0.5f, float maxAlpha= 1.5f) { 
  _changeAlpha    = val;
  _minAlpha       = minAlpha;
  _maxAlpha       = maxAlpha;
}
void MLP::setHeurChangeGain (bool val, float minGain = 0.5f, float maxGain = 2.0f) { 
  _changeGain    = val;
  _minGain       = minGain;
  _maxGain       = maxGain;
}
void MLP::setHeurShuffleDataset (bool val) { _shuffleDataset = val; }
void MLP::setHeurZeroWeights (bool val, float zeroThreshold) { 
  _zeroWeights   = val; 
  _zeroThreshold = zeroThreshold;
}

void MLP::setHeurTotalError (bool val) { _stopTotalError = val; }

void MLP::setHeurSelectWeights (bool val) { _selectWeights = val; }

int   MLP::getIterations () {
  return _iters;
}
int   MLP::getEpochs () {
  return _epochs;
}
int   MLP::getBatchSize () {
  return _batchSize;
}
float MLP::getAlpha () {
  return Alpha;
}
float MLP::getEta () {
  return Eta;
}
float MLP::getGain () {
  return Gain;
}
float MLP::getAnneal () {
  return _anneal;
}

int MLP::getNeuronNumbers (int layer) {
  if (layer >= _numLayers) return 0;
  return _units[layer];
}

float MLP::getWeight (int layer, int lower, int upper) {
  if (layer >= _numLayers) return 0;
  return Layer[layer]->Weight[upper][lower];
}

int MLP::setWeight (int layer, int upper, int lower, float val) { 
  if (layer >= _numLayers) return 0;
  if (upper > Layer[layer]->Units) return 0;
  if (lower > Layer[layer - 1]->Units) return 0;
  Layer[layer]->Weight[upper][lower] = val;
  return 1;
}

// Allocates the necessary memory for the network
void MLP::generateNetwork()
{
  Layer = (LAYER**) calloc(_numLayers, sizeof(LAYER*));

  for (int l = 0; l < _numLayers; l++) {
    Layer[l] = (LAYER*) malloc(sizeof(LAYER));

    Layer[l]->Number     = l;
    Layer[l]->Units      = _units[l];
    Layer[l]->Activation = _activations[l];
    Layer[l]->Output     = (float*)  calloc(_units[l] + 1, sizeof(float));
    Layer[l]->Error      = (float*)  calloc(_units[l] + 1, sizeof(float));
    Layer[l]->Weight     = (float**) calloc(_units[l] + 1, sizeof(float*));
    Layer[l]->WeightSave = (float**) calloc(_units[l] + 1, sizeof(float*));
    Layer[l]->dWeight    = (float**) calloc(_units[l] + 1, sizeof(float*));
    Layer[l]->dWeightOld = (float**) calloc(_units[l] + 1, sizeof(float*));
    Layer[l]->Output[0]  = BIAS;

    if (l != 0) {
      for (int i = 1; i <= _units[l]; i++) {
        Layer[l]->Weight[i]     = (float*) calloc(_units[l - 1] + 1, sizeof(float));
        Layer[l]->WeightSave[i] = (float*) calloc(_units[l - 1] + 1, sizeof(float));
        Layer[l]->dWeight[i]    = (float*) calloc(_units[l - 1] + 1, sizeof(float));
        Layer[l]->dWeightOld[i] = (float*) calloc(_units[l - 1] + 1, sizeof(float));
      }
    }
  }
  InputLayer  = Layer[0];
  OutputLayer = Layer[_numLayers - 1];
}

void MLP::changeEta () {
  static bool Eup = false;
  float eta = Eta;
  if (eta <= _minEta || eta >= _maxEta) Eup = !Eup;
  if (Eup) Eta = eta / _anneal;
  else Eta = eta * _anneal;
}

void MLP::changeGain () {
  static bool Gup = true;
  float gain = Gain;
  if (gain <= _minGain || gain >= _maxGain) Gup = !Gup;
  if (Gup) Gain = gain + 0.15;
  else Gain = gain - 0.15;
}

void MLP::changeAlpha () {
  static bool Aup = true;
  float alpha = Alpha;
  if (alpha <= _minAlpha || alpha >= _maxAlpha) Aup = !Aup;
  if (Aup) Alpha = alpha + 0.15;
  else Alpha = alpha - 0.15;
}

void MLP::changeBatchSize () {
  _batchSize /= 1.5;
  if (_batchSize > _nTrain / 4) _batchSize = _nTrain / 4;
  if (_batchSize < 1) _batchSize = 1;
}

// select the best set of weights over 10 random ones
void MLP::selectWeights(DATASET* dataset) {
  float minError = HUGE_VAL;
  // if (_verbose > 1) 
    Serial.println("Selecting best weights");
  for (uint8_t i = 0; i < 10; i++) {
    randomWeights (1.0f);
    float criterion = getTrainSetError (dataset);
    if (_stopTotalError) criterion += getTestSetError (dataset);
    if (criterion < minError) {
      minError = criterion;
      saveWeights();
    }
  }
  restoreWeights();
}

// Simple function to optimize the training of the network
float MLP::optimize(DATASET* dataset, int iters, int epochs, int batchSize)
{
  _iters = iters;
  _epochs = epochs;
  _batchSize = batchSize;
  bool Stop = false;
  uint16_t lastSave, lastIter = 0;
  int iter, epoch;
  _minError = MAX_float;

    // Estimate maximum training duration
  if (_verbose > 0) {
    uint32_t dur = estimateDuration (dataset);
    Serial.printf("Estimated duration for training and optimization: %u ms (%d min %d sec)\n", dur,
                dur / 60000, (dur % 60000) / 1000);
  }
  if (!_datasetProcessed) processDataset(dataset);

  if (_initialize) {
    if (_verbose > 0) Serial.println("Creating a new network");
    generateNetwork();
    (_selectWeights) ? selectWeights(dataset) : randomWeights(0.5f);
  }
  shuffleDataset (dataset, 0, _nData);


  for (iter = 0; iter < _iters; iter++) {
    /*
        Each iteration :
        - restore best weigths so far
        - according to heuristics parameters:
          - shuffle the complete dataset
          - change batchsize
          - mutate weights or create new random weights
        - run epochs
          - if too many epochs passed with no better solution:
            change eta & gain, shuffle training set
          - if a better solution is found: save the weights
          - if the error is lower than the min: exit
        Returns the final value of the error on the testing set
    */
    if (_verbose > 0) Serial.printf("\nIteration number %d", iter + 1);
    
    if (iter != 0) restoreWeights();
    if (iter - lastIter > 1) {
      lastIter = iter;

      if (_mutateWeights) {
        weightMutation (0.15f, 0.20f);
        if (_verbose > 0) Serial.print(" -> Weight mutation");
      }
      if (_changeWeights) {
        (_selectWeights) ? selectWeights(dataset) : randomWeights(1.0f);
        if (_verbose > 0) Serial.print(" -> New random weights");
      }
      if (_changeBatch) {
       changeBatchSize ();
        if (_verbose > 0) Serial.printf(", changing batch size to %d", _batchSize);
      }
    }
    if (_shuffleDataset) {
      shuffleDataset (dataset, 0, _nData); // Shuffle complete dataset
      if (_verbose > 1) Serial.print(", shuffling all dataset");
    }

    lastSave = 0;
    for (epoch = 0; epoch < _epochs; epoch++) {
      if (_verbose > 1) Serial.printf("\nEpoch %4d ", epoch);
      trainNet(dataset);
      testNet(dataset, true);

      // New best set of weights: save the weights
      if (_criterion < _minError) {
        if (_verbose > 1) Serial.print(" - saving network ...");
        lastIter = iter;
        _minError = _criterion;
        saveWeights();
        lastSave = epoch;
      }

      // Stop if the objective is reached
      if (_criterion < _maxErr) {
        if (_verbose > 1) Serial.println(" - stopping Training and restoring Weights ...");
        Stop = true;
        break;
      }

      // Heuristics: change the learning rate, the gain and shuffle training set
      if (epoch - lastSave >= _epochs / 10) {
        if (_changeEta)  changeEta  ();
        if (_changeGain) changeGain ();
        if (_changeAlpha) changeAlpha ();
        if (_shuffleDataset) {
          shuffleDataset (dataset, 0, _nTrain); // Shuffle training set
          if (_verbose > 1) Serial.print(" - Shuffling training set ...");
        }

        if (_verbose > 1 && (_changeGain || _changeEta)) 
          Serial.printf("\nChanging learning rate to %.3f, gain to %.2f, alpha to %.2f", Eta, Gain, Alpha);
        lastSave = epoch;
      }

    }
    if (Stop) {
      ++iter;
      break;
    }
  }
  restoreWeights();
  int nepochs = (iter - 1) * _epochs;
  if (nepochs < 0) nepochs = 0;
  _totalEpochs = nepochs + epoch;
  if (_verbose > 0) Serial.printf("\nFinished in %d iterations (%d epochs)\n", 
    iter, _totalEpochs);
  return _testError;
}

// Single forward and backward process
// void MLP::simulateNet(float* Input, float* Output, float* Target, bool BackProp)
// {
//   setInput(Input);
//   propagateNet();
//   getOutput(Output);
//   computeOutputError(Target);
//   if (BackProp) {
//     backpropagateNet();
//     adjustWeights();
//   }
// }

// Train on a batch of data
void MLP::trainNet(DATASET* dataset)
{
  float *Output;
  Output = new float [_units[_numLayers - 1]];
  if (!_datasetProcessed) processDataset(dataset);
  int sample = randomInt(0, _nTrain - _batchSize);
    process(&dataset->data[sample].In[0], Output,
                &dataset->data[sample].Out, _batchSize);
  // }
  delete Output;
}

float MLP::getTrainSetError (DATASET* dataset) {
  float *Output;
  Output = new float [_units[_numLayers - 1]];
  float trainError = 0;
  // _eval = true;
  for (int sample = 0; sample < _nTrain; sample++) {
    process(&dataset->data[sample].In[0], Output,
                &dataset->data[sample].Out, 1);
    trainError += Error;
  }
  _eval = false;
  delete Output;
  return trainError;
}

float MLP::getTestSetError (DATASET* dataset) {
  float *Output;
  Output = new float [_units[_numLayers - 1]];
  float testError = 0;
  // _eval = true;
  for (int sample = _nTrain; sample < _nData; sample++) {
    process(&dataset->data[sample].In[0], Output,
                &dataset->data[sample].Out, 1);
    testError += Error;
  }
  _eval = false;
  delete Output;
  return testError;
}



// Compute total error on training and testing sets
void MLP::testNet(DATASET* dataset, bool disp)
{
  _testError = getTestSetError (dataset);
  _criterion = _testError;
  if (_stopTotalError) {
    _trainError = getTrainSetError (dataset);
    _criterion += _trainError;
  }

  if (disp && _verbose > 0) {
    if (_criterion < _minError && _verbose == 1) Serial.println();
    if (_criterion < _minError || _verbose > 1) {
      if (!_stopTotalError) _trainError = getTrainSetError (dataset);
      // NMSE : Normalized Mean Square Error (cost function)
      Serial.printf("NMSE is %6.3f on Training Set and %6.3f on Test Set",
                    _trainError, _testError);
    }
  }
}

// Count the prediction errors on the training set and on the test set
void MLP::evaluateNet(DATASET* dataset, float threshold)
{
  float *Out;
  if (OutputLayer->Activation == SOFTMAX) Out = new float [OutputLayer->Units + 1];
  else Out = new float [_units[_numLayers - 1]];

  int nError = 0;
  // _eval = true;
  for (int sample = 0; sample < _nTrain; sample++) {
    process(&dataset->data[sample].In[0], Out, &dataset->data[sample].Out, 1);
    if (OutputLayer->Activation == SOFTMAX) {
      // Softmax case : look for the highest value
      int indexMax = 0;
      float valMax = 0;
      for (int j = 0; j < OutputLayer->Units; j++) {
        if (Out[j] > valMax) {
          valMax = Out[j];
          indexMax = j;
        }
      }
      if (abs(indexMax - dataset->data[sample].Out) > threshold) ++nError;
    } else {
      float x = Out[0] * _outDelta + _outMinVal;
      if (abs(x - dataset->data[sample].Out) > threshold) ++nError;
    }
  }
  if (_verbose > 0) Serial.printf("\nVerifying on %d train data : %2d errors (%.2f%%)\n", _nTrain, nError, 100.0 * nError / _nTrain);

  nError = 0;
  for (int sample = _nTrain; sample < _nData; sample++) {
    process(&dataset->data[sample].In[0], Out, &dataset->data[sample].Out, 1);
    if (OutputLayer->Activation == SOFTMAX) {
      // Softmax case : look for the highest value
      int indexMax = 0;
      float valMax = 0;
      for (int j = 0; j < OutputLayer->Units; j++) {
        if (Out[j] > valMax) {
          valMax = Out[j];
          indexMax = j;
        }
      }
      if (abs(indexMax - dataset->data[sample].Out) > threshold) ++nError;
    } else {
      float x = Out[0] * _outDelta + _outMinVal;
      if (abs(x - dataset->data[sample].Out) > threshold) ++nError;
    }
  }
  if (_verbose > 0) Serial.printf("Verifying on %d test data  : %2d errors (%.2f%%)\n", _nTest, nError, 100.0 * nError / _nTest);
  _eval = false;
  delete Out;
}

// Provide the estimated time of the complete training in ms
uint32_t MLP::estimateDuration (DATASET* dataset)
{
  saveWeights();
  unsigned long chrono = millis();
  trainNet(dataset);
  testNet(dataset, false);
  chrono = millis() - chrono;
  restoreWeights();
  return chrono * _iters * _epochs * 1.25f;
}

// Shuffle a portion of the dataset, in the provided range
void MLP::shuffleDataset (DATASET* dataset, int begin, int end)
{
  for (int i = 0; i < 2 * _nData; i++) {
    int ind1 = begin + random(end - begin);
    int ind2 = begin + random(end - begin);
    for (int j = 0; j < _units[0]; j++)
      SWAP(dataset->data[ind1].In[j], dataset->data[ind2].In[j]);
    SWAP(dataset->data[ind1].Out, dataset->data[ind2].Out);
  }
}

// // Forward propagation in a layer
// void MLP::propagateLayer(LAYER* Lower, LAYER* Upper)
// {
//     float Sum;
//     for (int i = 1; i <= Upper->Units; i++) {
//       Sum = 0;
//       for (int j = 0; j <= Lower->Units; j++)
//         Sum += Upper->Weight[i][j] * Lower->Output[j];
//       // Upper->Output[i] = 1 / (1 + exp(-Gain * Sum));
//       Upper->Output[i] = activation (Sum, Upper);
//     }
// }

// // Forward propagation in the network
// void MLP::propagateNet()
// {
//   for (int l = 0; l < _numLayers - 1; l++)
//     propagateLayer(Layer[l], Layer[l + 1]);
// }

// // Compute the error of the network after forward propagation
// void MLP::computeOutputError(float* Target)
// {
//   float Out, Err;
//     Error = 0;
//     for (int i = 1; i <= OutputLayer->Units; i++) {
//       Out = OutputLayer->Output[i];
//       // Err = Target[i - 1] - Out;
//       Err = (Target[i - 1] - _outMinVal) / _outDelta - Out;
//       // OutputLayer->Error[i] = Gain * Out * (1 - Out) * Err;
//       OutputLayer->Error[i] = derivActiv (Out, OutputLayer) * Err;
//       Error += 0.5 * Err * Err; // Cost function
//     }
// }

// // Back propagation of the error in a layer
// void MLP::backpropagateLayer(LAYER* Upper, LAYER* Lower)
// {
//   float Out, Err;
//   for (int i = 1; i <= Lower->Units; i++) {
//     Out = Lower->Output[i];
//     Err = 0;
//     for (int j = 1; j <= Upper->Units; j++)
//       Err += Upper->Weight[j][i] * Upper->Error[j];
//     // Lower->Error[i] = Gain * Out * (1 - Out) * Err;
//     Lower->Error[i] = derivActiv (Out, Upper) * Err;
//   }
// }

// // Back propagation of the error in the network
// void MLP::backpropagateNet()
// {
//   for (int l = _numLayers - 1; l > 1; l--)
//     backpropagateLayer(Layer[l], Layer[l - 1]);
// }

// // Compute the new weights after backpropagation
// void MLP::adjustWeights()
// {
//   float Out, Err, dWeight;
//   for (int l = 1; l < _numLayers; l++) {
//     for (int i = 1; i <= Layer[l]->Units; i++) {
//       for (int j = 0; j <= Layer[l - 1]->Units; j++) {
//         Out = Layer[l - 1]->Output[j];
//         Err = Layer[l]->Error[i];
//         dWeight = Layer[l]->dWeight[i][j];
//         Layer[l]->Weight[i][j] += Eta * Err * Out + Alpha * dWeight;
//         Layer[l]->dWeight[i][j] = Eta * Err * Out;

//         // Apply zero weights heuristics
//         if (_zeroWeights && abs(Layer[l]->Weight[i][j])<_zeroThreshold)
//           Layer[l]->Weight[i][j] = 0; 
//       }
//     }
//   }
// }

// Random initialization of the weights
void MLP::randomWeights(float x)
{
  x = abs(x);
  for (int l = 1; l < _numLayers; l++) {
    for (int i = 1; i <= Layer[l]->Units; i++){
      for (int j = 0; j <= Layer[l - 1]->Units; j++){
        Layer[l]->Weight[i][j] = randomFloat(-x, x);
      }
    }
  }
}

// Store the weights and parameters for later use
void MLP::saveWeights()
{
  EtaSave = Eta;
  GainSave = Gain;
  AlphaSave = Alpha;
  for (int l = 1; l < _numLayers; l++) {
    for (int i = 1; i <= Layer[l]->Units; i++) {
      for (int j = 0; j <= Layer[l - 1]->Units; j++)
        Layer[l]->WeightSave[i][j] = Layer[l]->Weight[i][j];
    }
  }
}

// Restore saved weights and parameters
void MLP::restoreWeights()
{
  Eta = EtaSave;
  Gain = GainSave;
  Alpha = AlphaSave;
  for (int l = 1; l < _numLayers; l++) {
    for (int i = 1; i <= Layer[l]->Units; i++) {
      for (int j = 0; j <= Layer[l - 1]->Units; j++)
        Layer[l]->Weight[i][j] = Layer[l]->WeightSave[i][j];
    }
  }
}

// Random integer in a range
int MLP::randomInt(int Low, int High)
{
  if (High < Low) return Low;
  return esp_random() % (High - Low + 1) + Low;
}

// Random float in a range
float MLP::randomFloat(float Low, float High)
{
  if (High < Low) return Low;
  return ((float) esp_random() / UINT32_MAX) * (High - Low) + Low;
}

void MLP::setInput(float* Input)
{
  for (int i = 1; i <= InputLayer->Units; i++)
    InputLayer->Output[i] = Input[i - 1];
}

void MLP::getOutput(float* Output)
{
  for (int i = 1; i <= OutputLayer->Units; i++)
    Output[i - 1] = OutputLayer->Output[i];
}

// Reads a positive integer in a file
int MLP::readIntFile (File file) {
  char buffer[10];
  uint8_t i = 0;
  while (file.available()) {
    char c = file.read();
    buffer[i++] = c;
    if (c == 10) return atoi(buffer); // CR
  }
}

// Reads a float from a file
float MLP::readFloatFile (File file) {
  char buffer[20];
  int i = 0;
  while (file.available()) {
    char c = file.read();
    buffer[i++] = c;
    if (c == 10) return atof(buffer); // CR
  }
}

float MLP::activation (float x, LAYER* layer)
{
  switch (layer->Activation) {
    case SIGMOID:
      if (Gain * x < -7) return 0;
      if (Gain * x >  7) return 1;
      return 1 / (1 + exp(-Gain * x));
      break;
    case SIGMOID2:
      return 1 - 2 / (1 + exp(Gain * x));
      break;
    case IDENTITY:
      return x;
      break;
    case RELU:
      return (x > 0) ? x : 0;
      break;
    case SELU:
      return (x > 0) ? 1.0507f * x : 1.0507f * 1.6732632f * (exp(x) - 1.0f);
      //               lambda        lambda      alpha
      break;
    case LEAKYRELU:
      return (x > 0) ? x : x * 0.1f;
      break;
    case ELU:
      return (x > 0) ? x : _alphaELU*(exp(x) - 1);
      break;
    case TANH:
      return tanh(x);
      break;
    case SOFTMAX:
      Serial.printf("Activ: Invalid activation function: %d\n", layer->Activation);   
      break;
    default:
      Serial.printf("Activ: Invalid activation function: %d\n", layer->Activation);
      while (1);
      break;
  }
}

float MLP::derivActiv (float x, LAYER* layer)
{
  switch (layer->Activation) {
    case SIGMOID:
      return Gain * x * (1 - x);
      break;
    case SIGMOID2:
      return Gain * (1 + x) * (1 - x) / 2.0f;
      break;
    case IDENTITY:
      return 1.0f;
      break;
    case RELU:
      return (x > 0) ? 1.0f : 0.0f;
      break;
    case SELU:
      return (x > 0) ? 1.0507f : 1.0507f * 1.6732632f + x; // x + lambda * alpha
      break;
    case LEAKYRELU:
      return (x > 0) ? 1.0f : 0.1f;
      break;
    case ELU: {
      int l = layer->Number;
      return (x > 0) ? 1.0f : _alphaELU + x;
      break;
      }
    case TANH:
      return 1 - x * x;
      break;
    case SOFTMAX:
      Serial.printf("Deriv: Layer %d Invalid activation function: %d\n", layer->Number, layer->Activation);
      break;
    default:
      Serial.printf("Deriv: Invalid activation function: %d\n", layer->Activation);
      while (1);
      break;
  }
}

/* Randomly slightly change the weights to try to move from local minimum
   Example : weightMutation (0.05, 0.10)
   Each weight has a 5% chance of being reduced or increased by up to 10%
*/
void MLP::weightMutation (float proba, float percent)
{
  for (int l = 1; l < _numLayers; l++)
    for (int i = 1; i <= Layer[l]->Units; i++)
      for (int j = 0; j <= Layer[l - 1]->Units; j++) {
        float p = randomFloat(0.f, 1.f);
        if (p < proba) Layer[l]->Weight[i][j] = Layer[l]->Weight[i][j] *
                            (1.0f + randomFloat(-percent, percent));
      }
}

void MLP::predict (float* Input, float *Output)
{
  _predict = true;
  process(Input, Output, Output, 1);
  if (OutputLayer->Activation != SOFTMAX) Output[0] = Output[0] * _outDelta + _outMinVal;
  _predict = false;
}

// Shift the input & output values of the dataset to the interval [0, 1]
void MLP::processDataset (DATASET* dataset)
{
  if (_datasetProcessed) return;
  if (_verbose > 0) Serial.println("Processing dataset");
  _datasetProcessed = true;
  
  // Input data
  for (int j = 0; j < dataset->nInput; j++) {
    _inMinVal[j] = HUGE_VAL;
    float _inMaxVal = -HUGE_VAL;
    for (int i = 0; i < dataset->nData; i++) {
      if (dataset->data[i].In[j] < _inMinVal[j]) _inMinVal[j] = dataset->data[i].In[j];
      if (dataset->data[i].In[j] > _inMaxVal) _inMaxVal = dataset->data[i].In[j];
    }
  _inDelta[j] = _inMaxVal - _inMinVal[j];
  }

  // Out data
  if (OutputLayer->Activation == SOFTMAX) { // No shift for Softmax
    _outMinVal = 0.;
    _outDelta = 1.;
    return;
  }

  float _outMaxVal = -HUGE_VAL;
  for (int i = 0; i < dataset->nData; i++) {
    if (dataset->data[i].Out < _outMinVal) _outMinVal = dataset->data[i].Out;
    if (dataset->data[i].Out > _outMaxVal) _outMaxVal = dataset->data[i].Out;
  }
  _outDelta = _outMaxVal - _outMinVal;

  if (OutputLayer->Activation == SIGMOID2 || OutputLayer->Activation == TANH) {
    _outMinVal += _outDelta / 2;
    _outDelta /= 2;
    // In this case shift the value to -1 .. 1
  }
  // to shift a value => value = (value - _outMinVal) / _outDelta
}

// Provide the values of errors on the training and testing sets
void MLP::getError (float *trainError, float *testError) {
  *trainError = _trainError;
  *testError = _testError;
}

int MLP::getTotalEpochs () { return _totalEpochs; }

// Softmax cost function for last layer
void MLP::softmax () {
  float *sum;
  int l = _numLayers - 1;
  sum = new float [OutputLayer->Units + 1];
  float denom = 0;
  float sumMax = -HUGE_VAL;
  for (int i = 1; i <= OutputLayer->Units; i++) {
    sum[i] = 0;
    for (int j = 0; j <= Layer[l - 1]->Units; j++) {
      sum[i] += Layer[l]->Weight[i][j] * Layer[l-1]->Output[j];
    }
    if (sum[i] > sumMax) sumMax = sum[i];
  }
  for (int i = 1; i <= OutputLayer->Units; i++) denom += exp(sum[i] - sumMax);
  for (int i = 1; i <= OutputLayer->Units; i++) {
    OutputLayer->Output[i] = exp(sum[i] - sumMax) / denom;
  }
  delete sum;
  return;
}

// This is the heart of the neural network : forward and back propagation
/*
    Input  : address of the input data of the mini batch 
    Output : provides the predicted value (end of forward propagation)
    Target : expected value (out of the dataset)
    batch  : size of the mini batch
*/
void MLP::process (float* Input, float* Output, float* Target, int batch) {
  // Comments use notations from https://en.wikipedia.org/wiki/Backpropagation 
  // Init batch
  for (int l = 1; l < _numLayers; l++) 
    for (int i = 1; i <= Layer[l]->Units; i++) 
      for (int j = 0; j <= Layer[l - 1]->Units; j++) {
        Layer[l]->dWeightOld[i][j] = Layer[l]->dWeight[i][j];
        Layer[l]->dWeight[i][j] = 0;
      }

// Process a batch of input data
  for (int iBatch = 0; iBatch < batch; iBatch++) {

  // Set input
    for (int i = 1; i <= InputLayer->Units; i++) {
      // InputLayer->Output[i] = Input[i - 1 + _units[0] * iBatch];
      InputLayer->Output[i] = (float) (Input[i - 1 + _units[0] * iBatch] - _inMinVal[i-1]) / _inDelta[i-1];
    }

  // Forward propagation
    for (int l = 0; l < _numLayers - 1; l++) {
      // Softmax case
      if (l == _numLayers - 2 && OutputLayer->Activation == SOFTMAX) {
        softmax();
      } else {
        // Other activation
        float Sum;
        for (int i = 1; i <= Layer[l + 1]->Units; i++) {
          Sum = 0;
          for (int j = 0; j <= Layer[l]->Units; j++) {
            Sum += Layer[l + 1]->Weight[i][j] * Layer[l]->Output[j];
          }
          Layer[l + 1]->Output[i] = activation (Sum, Layer[l + 1]);
        }
      }
    }

  // Get output
    for (int i = 1; i <= OutputLayer->Units; i++)
      Output[i - 1] = OutputLayer->Output[i];

  // Stop process if prediction
    if (_predict) return;

  // Output Error
    float Out, Err, Grad, Targ;
    Error = 0;
    for (int i = 1; i <= OutputLayer->Units; i++) {
      // Targ = (Target[i - 1 + (_units[_numLayers - 1] + 1) * iBatch] - _outMinVal) / _outDelta;
      if (OutputLayer->Activation == SOFTMAX) {
      // Cross entropy & softmax (see https://deepnotes.io/softmax-crossentropy)
        Out = OutputLayer->Output[i];
        Targ = 0;
        if (abs(Target[2 * iBatch] - (i - 1)) < 0.1) Targ = 1; // one-hot encoded
        Grad = Targ - Out; // Nabla a^L C
        OutputLayer->Error[i] = Grad; // delta^L
        Error -= Targ * log(Out + 0.00000001) / OutputLayer->Units;
      } else {
      // Squared Error loss
        Out = OutputLayer->Output[i];
        Targ = (Target[2 * iBatch] - _outMinVal) / _outDelta;
        Grad = Targ - Out; // Nabla a^L C
        OutputLayer->Error[i] = derivActiv (Out, OutputLayer) * Grad; // delta^L
        Error += 0.5 * Grad * Grad; // Cost function C  
      }
    }
    // if (_eval) return; // Stop process if evaluation

  // Backpropagate error
    for (int l = _numLayers - 1; l > 1; l--) {
      for (int i = 1; i <= Layer[l - 1]->Units; i++) {
        Out = Layer[l - 1]->Output[i];
        Err = 0;
        for (int j = 1; j <= Layer[l]->Units; j++)
          Err += Layer[l]->Weight[j][i] * Layer[l]->Error[j]; // W^l (T) . delta^l
        Layer[l - 1]->Error[i] = derivActiv (Out, Layer[l-1]) * Err; // delta^(l-1)
      }
    }
    
  // Adjust delta weights
    for (int l = 1; l < _numLayers; l++) 
      for (int i = 1; i <= Layer[l]->Units; i++) 
        for (int j = 0; j <= Layer[l - 1]->Units; j++) {
          Out = Layer[l - 1]->Output[j];              // a^(l-1) (T)
          Err = Layer[l]->Error[i];                   // delta^l
          Layer[l]->dWeight[i][j] += Eta * Err * Out; // eta * nabla W^l C
        }
  } // End for iBatch

  // Adjust weights
    for (int l = 1; l < _numLayers; l++) 
      for (int i = 1; i <= Layer[l]->Units; i++) 
        for (int j = 0; j <= Layer[l - 1]->Units; j++)
          Layer[l]->Weight[i][j] += (Layer[l]->dWeight[i][j]
           + Alpha * Layer[l]->dWeightOld[i][j]) / batch;

// End of process
}
