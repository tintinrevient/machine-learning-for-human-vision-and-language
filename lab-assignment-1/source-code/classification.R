library(keras)
library(numDeriv)

source("~/Documents/workspace/machine-learning-for-human-vision-and-language/lab-assignment-1/source-code/convolution_layer.R")
source("~/Documents/workspace/machine-learning-for-human-vision-and-language/lab-assignment-1/source-code/forward_propagation.R")
source("~/Documents/workspace/machine-learning-for-human-vision-and-language/lab-assignment-1/source-code/back_propagation.R")


#load and preprocess mnist data

# import the dataset
mnist <- dataset_mnist()
c(c(x_train, y_train), c(x_test, y_test)) %<-% mnist

# reshape from 3D to 2D
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))

# rescale from the interval [0,255] to [0,1]
x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)


#set up hyperparameters for the neural network

filter_1 <- array(round(runif(90, min=-3, max=3)), c(3,3,1,10))
filter_2 <- array(round(runif(180, min=-3, max=3)), c(3,3,10,10))
units <- 10
bias_vector_1 <- rep(0, dim(filter_1)[4])
bias_vector_2 <- rep(0, dim(filter_2)[4])
bias_vector_3 <- rep(0, units)

count <- 360
weight_matrix <- matrix(runif(count * units), nrow=count, ncol=units)
r <- 0.01

parameters <- list(filter_1, filter_2, weight_matrix, bias_vector_1, bias_vector_2, bias_vector_3)

#main loop

cycles <- 10
losslist <- array(data=0, dim=c(cycles))
i <- 0

for(x in seq(cycles))
{
  i <- i + 1
  input <- x_train[x,,,]
  input <- array_reshape(mnist_image, c(28, 28, 1))
  target <- y_train[x,]
  
  prediction <- forward_propagation(input, parameters[[1]], parameters[[2]], parameters[[3]], parameters[[4]], parameters[[5]], parameters[[6]], units)
  l <- loss(prediction, target)
  losslist[i] <- l
  
  newparameters <- back_propagation(input, parameters[[1]], parameters[[2]], parameters[[3]], parameters[[4]], parameters[[5]], parameters[[6]], units, target, r)
  parameters <- newparameters
  
}







