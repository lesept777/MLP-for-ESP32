# Classification of points in 4 square sectors

This example classifies points (x,y) in the domain [0,1] x [0,1] in 4 sectors:

```
/*    sectors:
       _______
      | 1 | 3 |
      | 0 | 2 |
       -------
*/
```
If x>0.5 and y<0.5, the output is 2.

This example uses the SIGMOID activation and the MSE cost function. The network is made of 3 layers, with :
* input layer: 2 neurons (x and y)
* hidden layer: 8 neurons
* output layer: 4 neurons (1 for each sector)

The dataset is made in the ino file, using the `sector` function
```
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
```
This function returns the sector's number (i.e. the output value) for a given (x,y) point.  `dataset.data[i].Out` contains the sector's number.

Standard execution will provide results such as:
* NMSE is  0.761 on Training Set and  0.080 on Test Set
* Verifying on 400 train data : 17 errors (4.25%)
* Verifying on 100 test data  :  4 errors (4.00%)

## Effect of regularization
If no regularization is used, the optimum set of weight found has the following properties:
* Average weight L1 norm: 6.87446 (lambda = 0.000000)
* Average weight L2 norm: 75.65907 (lambda = 0.000000)

Regularization adds a penalty to the cost function to try to reduce the values of the weights. L1 uses the sum of the absolute values of the weights, L2 uses the sum of the half of the squared values of the weights.

L1 or L2 regularizations can be forced by adding the following lines 
```
  Net.setHeurRegulL2 (true, 1); // for L2 regularization
  Net.setHeurRegulL1 (true, 5); // for L1 regularization
```
after setting the heuristics parameters:
```
Net.setHeuristics(heuristics);
```
The numbers (1 and 5) are floats parameters that set the importance of the regularization in the cost function. They are divided by 1000000 in the code, and appear as such in the network summary if you display it using
```
  Net.displayNetwork();
```

## L1 regularization
```
  Net.setHeurRegulL1 (true, 10); // for L1 regularization
```
provide the following results (you may obtain different results as the process is randomly initialized):
* NMSE is  1.466 on Training Set and  0.154 on Test Set
* Verifying on 400 train data : 18 errors (4.50%)
* Verifying on 100 test data  :  3 errors (3.00%)
* Average weight L1 norm: 3.68611 (lambda = 0.000010)
* Average weight L2 norm: 49.88160 (lambda = 0.000000)

The training and test performances are a little bit degraded but prediction remains correct (no error in 20 random cases). The average L1 norm of the weight was reduced.

Setting the L1 parameter to `100` leads to less good results:
* NMSE is  3.725 on Training Set and  0.762 on Test Set
* Verifying on 400 train data : 72 errors (18.00%)
* Verifying on 100 test data  : 16 errors (16.00%)

and a few prediction errors (2 in 20) but much lower weights:
* Average weight L1 norm: 1.00118 (lambda = 0.000100)
* Average weight L2 norm: 3.24623 (lambda = 0.000000)


## L2 regularization
```
  Net.setHeurRegulL2 (true, 1); // for L2 regularization
```
provide the following results:
* NMSE is  1.429 on Training Set and  0.226 on Test Set

The results are worse here:
* Verifying on 400 train data : 208 errors (52.00%)
* Verifying on 100 test data  : 59 errors (59.00%)

Prediction error is quite high (4 errors in 20 tests) and the norms of the weights are
* Average weight L1 norm: 0.82061 (lambda = 0.000000)
* Average weight L2 norm: 0.50392 (lambda = 0.000001)

Clearly, the impact of regularization is too high. Setting the parameter to `0.9` leads to better results:
* NMSE is  0.985 on Training Set and  0.327 on Test Set
* Verifying on 400 train data : 17 errors (4.25%)
* Verifying on 100 test data  : 11 errors (11.00%)

and no prediction error but sensibly lower weights:
* Average weight L1 norm: 3.70065 (lambda = 0.000e+00)
* Average weight L2 norm: 25.12603 (lambda = 9.000e-07)

L2 regularization seems to be more sensitive to the value of its parameter.