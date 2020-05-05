/*
    Multilayer Perceptron library for ESP32
    Inspired by:
    https://courses.cs.washington.edu/courses/cse599/01wi/admin/Assignments/bpn.html
    http://neuralnetworksanddeeplearning.com/chap2.html
*/

#ifndef MLP_h
#define MLP_h

#include <Arduino.h>
#include "FS.h"
#include "SPIFFS.h"

#define MAX_LAYERS    8

// Heuristics options: set them if...
#define H_INIT_OPTIM   0x01  // if you initialize optimize
#define H_CHAN_WEIGH   0x02  // for brand new random weights
#define H_MUTA_WEIGH   0x04  // to slightly change the weights
#define H_CHAN_BATCH   0x08  // to change natch size
#define H_CHAN_LRATE   0x10  // to change the learning rate
#define H_CHAN_SGAIN   0x20  // to change the sigmoid gain
#define H_CHAN_ALPHA   0x40  // to change the sigmoid gain
#define H_SHUF_DATAS   0x80  // to shuffle the dataset
#define H_ZERO_WEIGH   0x100  // to force low weights to 0


// Activation functions
enum ACTIVATION {
  SIGMOID,
  SIGMOID2,  /* Sigmoid like function between -1 & 1 */
  IDENTITY,
  RELU,
  TANH,
  SOFTMAX
};

typedef struct {           /* A layer of the network:                 */
  int       Number;        /* - number of the layer in the network    */
  int       Units;         /* - number of neurons in this layer       */
  int       Activation;    /* - the number of the activation function */
  float*    Output;        /* - output of ith neuron                  */
  float*    Error;         /* - error term of ith neuron              */
  float**   Weight;        /* - connection weights to ith neuron      */
  float**   WeightSave;    /* - saved weights for stopped training    */
  float**   dWeight;       /* - last weight deltas for momentum       */
} LAYER;

typedef struct
{
  float *In;	  // dynamic array of input data
  float Out;	  // output
} Data;

typedef struct
{
  Data *data;	  // dynamic array of data
  size_t nData;	  // number of data
  size_t nInput;  // number of input (In)
} DATASET;


class MLP
{
  public:
    MLP(int, int units[], int verbose = 1);
    ~MLP();

/*
    Functions related to dataset and network saving on SPIFFS
*/
// netLoad (filename), netSave (filename)
    bool  netLoad (const char* const);
    void  netSave (const char* const);
/*
    readCsvFromSpiffs (filename, dataset, nData, coeff)
    Reads the dataset from a cdv file on SPIFFS
    nData : number of lines of the file
    A line is made of: x1, x2, x3 ... xN, Out
    where N is the number of neurons of the input layer
    coeff : a coefficient to divide the out values if they are too big
*/
    int   readCsvFromSpiffs (const char* const, DATASET*, int, float);
/*
    createDataset (dataset, nData)
    nData: number of data in the dataset
    A data is made of: x1, x2, x3 ... xN, Out
    where N is the number of neurons of the input layer
    Keep 'out' near the range 0 .. 10
*/
    int   createDataset (DATASET*, int);
// process the dataset (minimum value and range)
    void  processDataset(DATASET*);
    void  destroyDataset(DATASET*);
// Display the network parameters
    void  displayNetwork();

/*
    The following methods are required
    begin (ratio): ratio of the training data in the complete dataset
    initLearn (alpha, eta, gain, anneal)
        4 parameters of the training phase:
        alpha : initial momentum
        eta   : initial learning rate
        gain  : initial gain of the sigmoid activation function
        anneal: rate of change of the learning rate
*/
    void  begin (float);
    void  initLearn (float, float, float, float);

/*
    Methods to set various parameters, if you need to change
    them anytime during the training phase
*/
    void  setIterations (int);
    void  setEpochs (int);
    void  setBatchSize (int);
    void  setAlpha (float);
    void  setEta (float);
    void  setGain (float);
    void  setAnneal (float);
    void  setActivation (int activation[]);
/*
    set the verbose level
    0: mute
    1: very few information (optimal)
    2: show all steps
    3: level 2 plus display weight values (in displayNetwork and netLoad)
*/
    void  setVerbose (int);

/*
    Methods to get various parameters
*/
    int   getIterations ();
    int   getEpochs ();
    int   getBatchSize ();
    float getAlpha ();
    float getEta ();
    float getGain ();
    float getAnneal ();
// getNeuronNumbers (layer): get the number of neurons in a specific layer
    int   getNeuronNumbers (int);
/*
    getWeight(layer, upperNeuron, lowerNeuron): get the value of a 
    specific weight
    setWeight(layer, upperNeuron, lowerNeuron, value): set any weight value
*/
    float getWeight (int, int, int);
    int  setWeight (int, int, int, float);
// Allocates memory for the network
    void  generateNetwork ();

/*
    The optimize method is a training solution
    Parameters:
    - dataset
    - number of iterations
    - number of epochs per iteration
    - number of samples in a batch
    Set the heuristics options before calling 'optimize'
    otherwise default options are taken
*/
    float optimize (DATASET*, int, int, int);
/*
    Define the various heuristics used in the 'optimize' method
    A heuristics is defined as an integer whose bits indicate
    various options (see the #define)
*/
    void  setHeuristics (int);
/*
    Methods to set the options one by one
    bool : true / false to allow or disable
    float, float: set a range (minimum and maximum values)
*/
    void  setHeurInitialize (bool);
    void  setHeurChangeWeights (bool, float);
    void  setHeurMutateWeights (bool, float, float);
    void  setHeurChangeBatch (bool);
    void  setHeurChangeEta (bool, float, float);
    void  setHeurChangeGain (bool, float, float);
    void  setHeurChangeAlpha (bool, float, float);
    void  setHeurShuffleDataset (bool);
    void  setHeurZeroWeights (bool, float);
//  Display the summary of the heuristics options
    void  displayHeuristics ();
//  Methods to force the change of the Alpha, Eta Gain and Batchsize values
    void  changeEta ();
    void  changeGain ();
    void  changeAlpha ();
    void  changeBatchSize ();
// setMaxError (maxError): set the criterium for stopping the learning phase
    void  setMaxError (float);
/*
    If you want to program your own optimization function, use the following
    methods
    trainNet: does the complete propagation - backpropagation
     - weight update process
    testNet: computes the current error in the training and testing sets
    getError: returns the values of the errors
*/
    void  trainNet (DATASET*);
    void  testNet (DATASET*, bool);
    void  evaluateNet (DATASET*, float);
    void  getError (float*, float*);

/*
    Once the net in trained and optimized, use the predict method
    for the inference
    Parameters:
    input: a pointer to the array of input data (in the format of the dataset)
    output: a pointer to the array of output result
*/
    void  predict (float*, float*);

/*
    Various useful functions
    estimateDuration: provide an estimate of the duration (in ms)
    requires that the dataset is created, the number of iterations
    and epochs are set

    randomWeights: affect random values to the weights (parameter: range)
    weightMutation: slightly change the weights
    parameters: probability of change, range of change (in %)

    saveWeights and restoreWeights: useful when a good error level is reached
*/
    uint32_t estimateDuration (DATASET*);
    void shuffleDataset (DATASET*, int, int);
    void randomWeights(float);
    void saveWeights();
    void restoreWeights();
    void weightMutation (float, float);

  private:

    // Parameters of the network
    LAYER**  Layer;         /* - layers of this net           */
    LAYER*   InputLayer;    /* - input layer                  */
    LAYER*   OutputLayer;   /* - output layer                 */
    float    Alpha;         /* - momentum factor              */
    float    Eta;           /* - learning rate                */
    float    Gain;          /* - gain of sigmoid function     */
    float    Error;         /* - total net error              */
    float    AlphaSave;       /* - saved learning rate          */
    float    EtaSave;       /* - saved learning rate          */
    float    GainSave;      /* - saved gain                   */

    // Private variables
    int      _units[MAX_LAYERS], _numLayers;
    int      _activations[MAX_LAYERS] = {0};
    int      _nData, _nTrain, _nTest;
    float    _ratio = 0.8f;
    int      _iters, _epochs, _batchSize;
    float    _anneal = 0.8f;
    float    _trainError, _testError;
    float    _maxErr = 0.05f;
    uint8_t  _verbose;
    float    _minTestError;
    bool     _datasetProcessed = false;
    float    _minVal, _delta;
    char     ActivNames[6][10] = {"SIGMOID", "SIGMOID2", "IDENTITY", 
                                  "RELU", "TANH", "SOFTMAX"
                                 };
    // Booleans for the heuristics
    int      _heuristics     = 0;
    bool     _initialize     = true;
    bool     _changeWeights  = false;
    bool     _mutateWeights  = false;
    bool     _changeBatch    = false;
    bool     _changeEta      = false;
    bool     _changeGain     = false;
    bool     _shuffleDataset = false;
    bool     _zeroWeights    = false;
    bool     _changeAlpha    = false;

    float    _proba = 0.05f, _percent = 0.15f;
    float    _range = 1.0f;
    float    _minEta = 0.35f,  _maxEta = 1.1f;
    float    _minGain = 0.5f,  _maxGain = 2.0f;
    float    _minAlpha = 0.5f, _maxAlpha = 1.5f;
    float    _zeroThreshold = 0.1f;

    // Private methods
    void  simulateNet(float*, float*, float*, bool);
    void  propagateLayer(LAYER*, LAYER*);
    void  propagateNet();
    void  computeOutputError(float*);
    void  backpropagateLayer(LAYER*, LAYER*);
    void  backpropagateNet();
    void  setInput(float*);
    void  getOutput(float*);
    void  adjustWeights();
    int   randomInt(int, int);
    float randomFloat(float, float);
    int   readIntFile (File);
    float readFloatFile (File);
    float activation (float, LAYER*, int);
    float derivActiv (float, LAYER*, int);
    float softmax (int, int);
};

#endif
