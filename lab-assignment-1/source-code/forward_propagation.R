library(keras)

source("~/Documents/workspace/machine-learning-for-human-vision-and-language/lab-assignment-1/source-code/convolution.R")
source("~/Documents/workspace/machine-learning-for-human-vision-and-language/lab-assignment-1/source-code/max_pooling.R")
source("~/Documents/workspace/machine-learning-for-human-vision-and-language/lab-assignment-1/source-code/normalisation.R")
source("~/Documents/workspace/machine-learning-for-human-vision-and-language/lab-assignment-1/source-code/relu.R")
source("~/Documents/workspace/machine-learning-for-human-vision-and-language/lab-assignment-1/source-code/softmax.R")
source("~/Documents/workspace/machine-learning-for-human-vision-and-language/lab-assignment-1/source-code/flatten.R")
source("~/Documents/workspace/machine-learning-for-human-vision-and-language/lab-assignment-1/source-code/fully_connected_layer.R")

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

mnist_filters <- array(round(runif(18, min=-3, max=3)),c(3,3,2))

layer_core <- function(filters, input, withMaxPooling) {
  
  #convolutional layer
  featuremaps <- convolutionOperation(filters, input)
  featuremaps <- relu(featuremaps)
  
  if (withMaxPooling){
    stacks <- array(0, dim=c(ceiling(dim(featuremaps)[1]/2), ceiling(dim(featuremaps)[2]/2), dim(featuremaps)[3]))
  }
  else {
    stacks <- array(0, dim=c(dim(featuremaps)[1], dim(featuremaps)[2], dim(featuremaps)[3]))
  }
  
  # max pooling + normaliztion
  for(i in seq(dim(featuremaps)[3])) {
    
    featuremap <- featuremaps[,,i]
    
    if(withMaxPooling) {
      featuremap <- maxPooling(featuremap, 2, 2)
    }

    stacks[,,i] <- normalisation(featuremap)
  }
  
  #output
  stacks
}

forward_propagation <- function(filter1, filter2, weight_matrix, input, units) {
  intermediate_output <- layer_core(filter1, input, TRUE)
  output <- layer_core(filter2, intermediate_output, FALSE)
  
  flatten_output <- flatten(output)
  dense_output <- denseLayer(flatten_output, units, weight_matrix)
  softmax_output <- softmax(dense_output)
  
  #output
  softmax_output
}


#testing
mnist_image <- x_train[1,,,]
mnist_image <- array_reshape(mnist_image, c(28, 28, 1))
mnist_filter_1 <- array(round(runif(18, min=-3, max=3)), c(3,3,1,2))
mnist_filter_2 <- array(round(runif(18, min=-3, max=3)), c(3,3,2,2))
units = 10
input_count = ceiling((dim(mnist_image)[1]-2)/2 - 2) * ceiling((dim(mnist_image)[2]-2)/2 - 2) * 2
weight_matrix <- matrix(runif(input_count * units), nrow=input_count, ncol=units)

output <- forward_propagation(mnist_filter_1, mnist_filter_2, weight_matrix, mnist_image, 10)

