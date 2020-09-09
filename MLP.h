/*
    Multilayer Perceptron library for ESP32
    Inspired by:
    https://courses.cs.washington.edu/courses/cse599/01wi/admin/Assignments/bpn.html
    http://neuralnetworksanddeeplearning.com/chap2.html

    (c) 2020 Lesept
    contact: lesept777@gmail.com

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
    OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
    ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
    OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef MLP_h
#define MLP_h

#include <Arduino.h>
#include "FS.h"
#include "SPIFFS.h"

#define MAX_LAYERS    8      // Maximum number of layers
#ifndef MAX_INPUT
#define MAX_INPUT    40      // Maximum number of neurons in input layer
#endif

// Heuristics options: set them if...
#define H_INIT_OPTIM     0x01  // if you initialize optimize
#define H_CHAN_WEIGH     0x02  // for brand new random weights
#define H_MUTA_WEIGH     0x04  // to slightly change the weights
#define H_CHAN_BATCH     0x08  // to change natch size
#define H_CHAN_LRATE     0x10  // to change the learning rate
#define H_CHAN_SGAIN     0x20  // to change the sigmoid gain
#define H_CHAN_ALPHA     0x40  // to change the sigmoid gain
#define H_SHUF_DATAS     0x80  // to shuffle the dataset
#define H_ZERO_WEIGH    0x100  // to force low weights to 0
#define H_STOP_TOTER    0x200  // stop optimization if test + train Error < threshold 
#define H_SELE_WEIGH    0x400  // select best weights over 10 random sets
#define H_FORC_S_G_D    0x800  // force stochastic gradient descent for faster optimization
#define H_REG1_WEIGH   0x1000  // use L1 weight regularization
#define H_REG2_WEIGH   0x2000  // use L2 weight regularization


// Activation functions
enum ACTIVATION {
  SIGMOID,
  SIGMOID2,     /* Sigmoid like function between -1 & 1 */
  IDENTITY,
  RELU,
  LEAKYRELU,    /* RELU with a small slope for negative values */
  ELU,          /* Similar to SIGMOID2 for <0, and RELU for >0 */
  SELU,
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
  float**   dWeight;       /* - weight deltas                         */
  float**   dWeightOld;    /* - last weight deltas for momentum       */
} LAYER;

typedef struct
{
  float *In;      // dynamic array of input data
  float Out;      // output
} Data;

typedef struct
{
  Data *data;     // dynamic array of data
  size_t nData;   // number of data
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
    int   setWeight (int, int, int, float);
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
    void  setHeuristics (long);
/*
    Methods to set the options one by one
    bool : true / false to allow or disable
    float, float: set a range (minimum and maximum values) or probability
*/
    void  setHeurInitialize (bool);
    void  setHeurZeroWeights (bool, float);
    // first float is the weight range, second is the probability to set weight to 0 (for sparsity)
    void  setHeurChangeWeights (bool, float, float);
    // first float is the mutation probability, second is the percent of change
    void  setHeurMutateWeights (bool, float, float);
    void  setHeurChangeBatch (bool);
    // in the following methods, the float arguments are min and max values of the range
    void  setHeurChangeEta (bool, float, float);
    void  setHeurChangeGain (bool, float, float);
    void  setHeurChangeAlpha (bool, float, float);
    void  setHeurShuffleDataset (bool);
    void  setHeurTotalError (bool);
    void  setHeurSelectWeights (bool);
    // in the 2 following methods, the float argument is the value of lambda (regul parameter)
    void  setHeurRegulL1 (bool, float);
    void  setHeurRegulL2 (bool, float);
//  Display the summary of the heuristics options
    void  displayHeuristics ();
//  Methods to force the change of the Alpha, Eta Gain and Batchsize values
    void  changeEta ();
    void  changeGain ();
    void  changeAlpha ();
    void  changeBatchSize ();
// setMaxError (maxError): set the criterium for stopping the learning phase
    void  setMaxError (float);

// select the best set of weights over 20 random ones
    void  selectWeights(DATASET*);
// parameters for regularization (L1 & L2)
    float regulL1Weights();
    float regulL2Weights();
    int   numberOfWeights();
/*
    If you want to program your own optimization function, use the following
    methods
    trainNet: does the complete propagation + backpropagation
     + weight update process
    testNet: computes the current error in the training and testing sets
    getError: returns the values of the errors
*/
    void  trainNetSGD (DATASET*);
    void  testNet (DATASET*, bool);
    void  trainAndTest (DATASET*);
    void  evaluateNet (DATASET*, float);
    void  getError (float*, float*);
    float getTrainSetError (DATASET*);
    float getTestSetError (DATASET*);
    int   getTotalEpochs ();

/*
    Once the net in trained and optimized, use the predict method
    for the inference
    Parameters:
    input: a pointer to the array of input data (in the format of the dataset)
    output: a pointer to the array of output result
*/
    // void  predict (float*, float*);
    float  predict (float*);

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



    void displayData (DATASET*);
    void disp(float*, float*, int, int);


  private:

    // Parameters of the network
    LAYER**  Layer;         /* - layers of this net           */
    LAYER*   InputLayer;    /* - input layer                  */
    LAYER*   OutputLayer;   /* - output layer                 */
    float    Alpha;         /* - momentum factor              */
    float    Eta;           /* - learning rate                */
    float    Gain;          /* - gain of sigmoid function     */
    float    Error;         /* - total net error              */
    float    AlphaSave;     /* - saved learning rate          */
    float    EtaSave;       /* - saved learning rate          */
    float    GainSave;      /* - saved gain                   */

    // Private variables
    int      _units[MAX_LAYERS], _numLayers;
    int      _activations[MAX_LAYERS] = {0};
    int      _nData, _nTrain, _nTest;
    float    _ratio = 0.8f;
    int      _iters, _epochs, _batchSize;
    float    _anneal = 0.8f;
    float    _trainError, _testError, _criterion;
    float    _maxErr = 0.05f;
    uint8_t  _verbose;
    float    _minError;
    bool     _datasetProcessed = false;
    float    _inMinVal[MAX_INPUT], _inDelta[MAX_INPUT];
    float    _outMinVal, _outDelta;
    float    _alphaELU = 1.0f;
    float    _lambdaRegulL1 = 0.0f;
    float    _lambdaRegulL2 = 0.0f;
    char     ActivNames[9][10] = {"SIGMOID", "SIGMOID2", "IDENTITY", 
                                  "RELU", "LEAKYRELU", "ELU", "SELU",
                                  "TANH", "SOFTMAX"};
    // Booleans for the heuristics
    long     _heuristics     = 0;
    bool     _initialize     = true;
    bool     _changeWeights  = false;
    bool     _mutateWeights  = false;
    bool     _changeBatch    = false;
    bool     _changeEta      = false;
    bool     _changeGain     = false;
    bool     _shuffleDataset = false;
    bool     _zeroWeights    = false;
    bool     _changeAlpha    = false;
    bool     _stopTotalError = false;
    bool     _selectWeights  = false;
    bool     _forceSGD       = false;
    bool     _regulL1        = false;
    bool     _regulL2        = false;

    float    _proba = 0.05f, _percent = 0.15f;
    float    _range = 1.0f;
    float    _minEta = 0.35f,  _maxEta = 1.1f;
    float    _minGain = 0.5f,  _maxGain = 2.0f;
    float    _minAlpha = 0.5f, _maxAlpha = 1.5f;
    float    _zeroThreshold = 0.1f;
    int      _totalEpochs;
    bool     _eval = false, _predict = false;
    float    _probaZeroWeight = 0.0f;

    // Private methods
    void  simulateNet(float*, float*, float*, bool);
    void  process(float*, float*, float*, int);
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
    float activation (float, LAYER*);
    float derivActiv (float, LAYER*);
    void  softmax ();
};

#endif