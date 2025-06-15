# Batch Normalization

## Use case
Often used in convolution to address the issue of covariate shift, making the training more stable.

What is covariate shift? It means the distribution of input features (covariates) differs between the training and test datasets, but the relationship between the input features and the target variable remains the same.

What is the issue? A big change of layer input -> a big change of the layer output -> a big change of the loss -> a big change of the gradient -> a big change of the network weight -> slow learning.

## Definition
Get the distribution statistic of each row and column, mean and variance. Subtracted by the mean, divided by square root of the variance. This will result in a Gaussian distribution with mean=0, variance=1. 

## Pros and Cons
### pros
* TBD 

### cons
* each batch has its own combination of different layers (a row for cat, a row for dog etc.), which result in different mean and variance. If the mean changes a lot among different batches, the covariate shift would still exist.
* to avoid the unstable means, we are forced to use a very large batch size.


## Heuristics
To address the issue of shift of mean (especially with a small batch size), layer normalization can be used as an alternative, which gets mean and variance along each layer (a row for a single item, e.g. dog or cat).

## Minimal Implementation
* PyTorch
TBD.

* TensorFlow
TBD.

## Reference
* https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
