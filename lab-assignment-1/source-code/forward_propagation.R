library(keras)

source("~/Documents/workspace/machine-learning-for-human-vision-and-language/lab-assignment-1/source-code/convolution_layer.R")
source("~/Documents/workspace/machine-learning-for-human-vision-and-language/lab-assignment-1/source-code/relu.R")
source("~/Documents/workspace/machine-learning-for-human-vision-and-language/lab-assignment-1/source-code/max_pooling.R")
source("~/Documents/workspace/machine-learning-for-human-vision-and-language/lab-assignment-1/source-code/normalisation.R")
source("~/Documents/workspace/machine-learning-for-human-vision-and-language/lab-assignment-1/source-code/flatten.R")
source("~/Documents/workspace/machine-learning-for-human-vision-and-language/lab-assignment-1/source-code/dense_layer.R")
source("~/Documents/workspace/machine-learning-for-human-vision-and-language/lab-assignment-1/source-code/softmax.R")

# import the dataset
mnist <- dataset_mnist()
c(c(x_train, y_train), c(x_test, y_test)) %<-% mnist

# reshape from 3D to 2D
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))

# rescale from the interval [0,255] to [0,1]
x_train <- x_train / 255
x_test <- x_test / 255

# rescale from the interval [0, 9] to [0, 1]
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

deep_layer <- function(input, filter, bias_vector, with_max_pooling) {
  
  stacks <- convolution_layer(input, filter, bias_vector)
  stacks <- relu(stacks)
  
  if (with_max_pooling){
    output <- array(0, dim=c(ceiling(dim(stacks)[1]/2), ceiling(dim(stacks)[2]/2), dim(stacks)[3]))
    output <- max_pooling(stacks, 2, 2)
  }
  else {
    output <- stacks
  }
  
  output <- normalisation(output)
  
  #return the output
  output
}

forward_propagation <- function(input, filter_1, filter_2, weight_matrix, bias_vector_1, bias_vector_2, bias_vector_3, units) {
  deep_1 <- deep_layer(input, filter_1, bias_vector_1, TRUE)
  deep_2 <- deep_layer(deep_1, filter_2, bias_vector_2, FALSE)
  
  flatten_1 <- flatten(deep_2)
  dense_1 <- dense_layer(flatten_1, units, weight_matrix, bias_vector_3)
  softmax_1 <- softmax(dense_1)
  
  #output
  softmax_1
}

#testing
input <- x_train[1,,,]
input <- array_reshape(mnist_image, c(28, 28, 1))
filter_1 <- array(round(runif(18, min=-3, max=3)), c(3,3,1,2))
filter_2 <- array(round(runif(18, min=-3, max=3)), c(3,3,2,2))
units <- 10
bias_vector_1 <- rep(0, dim(filter_1)[4])
bias_vector_2 <- rep(0, dim(filter_2)[4])
bias_vector_3 <- rep(0, units)

count = ceiling((dim(input)[1]-2)/2 - 2) * ceiling((dim(input)[2]-2)/2 - 2) * 2
weight_matrix <- matrix(runif(count * units), nrow=count, ncol=units)

output <- forward_propagation(input, filter_1, filter_2, weight_matrix, bias_vector_1, bias_vector_2, bias_vector_3, units)

