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
The numbers (1 and 5) are parameters that set the importance of the regularization in the cost function.

L1 regularization using
```
  Net.setHeurRegulL1 (true, 10); // for L1 regularization
```
provide the following results:
