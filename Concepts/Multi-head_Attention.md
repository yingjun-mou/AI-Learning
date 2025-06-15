# Multi-head Attention

## Use case
TBD

## Definition
Attention: each embedding (of a subset of the doc) is a weighted sum (weighted by attention score of each token).

## Pros and Cons
### pros
* TBD 

### cons
* TBD


## Heuristics
* Attention mask: mask out token comes after a target token, by overwriting it to -inf, so that the softmax value will be 0.

## Minimal Implementation
* PyTorch
TBD.

* TensorFlow
TBD.

## Reference
* TBD
