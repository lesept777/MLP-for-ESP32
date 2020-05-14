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

This example uses the SOFTMAX activation and the CROSS ENTROPY cost function. The network is made of 3 layers, with :
* input layer: 2 neurons (x and y)
* hidden layer: 8 neurons
* output layer: 4 neurons (1 for each sector)

The dataset is made in the ino file, using the `sector` function
```
int sector (float x, float y) {
  if (x <  0.5 && y < 0.5)  return 0;
  if (x <  0.5 && y >= 0.5) return 1;
  if (x >= 0.5 && y < 0.5)  return 2;
  if (x >= 0.5 && y >= 0.5) return 3;
}
```
This function returns the sector's number (i.e. the output value) for a given (x,y) point.  `dataset.data[i].Out` contains the sector's number.

3 output results are provided, in 3 files:
* Output.txt: standard optimization, using `SIGMOID` activation for the hidden layer
* OtherOutput.txt: `LEAKY RELU` activation for the hidden layer
* OtherCriterion.txt: `SIGMOID` activation for the hidden layer and additional `H_STOP_TOTER` criterion in the heuristics. With this option, the stopping criterion for the heuristics is based on the error on the total dataset, not only on the test subset.

This last example provides a total NMSE under 1, and no error in the validation (done with 20 random points) while the other results give 1 and 3 errors.