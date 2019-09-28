#prerequisite: install.packages("numDeriv")
library(numDeriv)
library(keras)

#L2 loss function based on Euclidian distance between two vectors
loss_function <- function(target, output)
{
  result <- (target - output)^2
  
  #return the result
  sum(result)/length(result)
}

#calculate the mean square error on a batch of dataset, e.g., y_train[1:10,] = batch_of_targets
#the batch can be the entire dataset or parts of dataset or just a single example
#stochastic gradient descent (SGD) uses only a single example (a batch size of 1) per iteration
#mini-batch stochastic gradient descent (mini-batch SGD) typically uses between 10 and 1,000 examples, chosen at random per iteration
mean_square_error <- function(batch_of_targets, batch_of_outputs)
{
  count <- dim(batch_of_targets)[1]
  
  #apply(x, 1, sum) -> compute the sum row by row
  #apply(x, 2, sum) -> compute the sum column by column
  result <- apply((batch_of_targets - batch_of_outputs)^2, 2, sum)/count
  
  #return the result
  sum(result)/length(result)
}

convolution_layer_gradient_descent <- function()
{
  loss_function()
}
