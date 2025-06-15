# RMS Mean

## Use case
Used for replacing layer normalization. Layer normalization addresses covariate shifts by using the mean and std of each row. In the "Root Mean Square Layer Normalization" paper, it points that the benefits of layer normalization is not mainly from its re-centering invariant, but its re-scaling invariant. In anther word, we don't care whether the mean of each row is zero, as long as the mean of each row is stable around a specific value. Thus, we don't need to compute mean and the computation can be optimized.

## Definition
"Root Mean Square mean". Equals to the square root of the mean of the element squared.

## Pros and Cons
### pros
* TBD 

### cons
* TBD


## Heuristics
* TBD

## Minimal Implementation
* PyTorch
TBD.

* TensorFlow
TBD.

## Reference
* TBD
