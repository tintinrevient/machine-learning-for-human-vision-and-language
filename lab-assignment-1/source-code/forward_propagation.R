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
  deep_2 <- deep_layer(deep_1, filter_2, bias_vector_2, TRUE)
  
  flatten_1 <- flatten(deep_2)
  dense_1 <- dense_layer(flatten_1, units, weight_matrix, bias_vector_3)
  softmax_1 <- softmax(dense_1)
  
  #output
  softmax_1
}

#testing
#input <- x_train[1,,,]
#input_depth <- 1
#input <- array_reshape(mnist_image, c(28, 28, input_depth))

#The 3rd dimension of filter_1 should be equal to the 3rd dimension of input
#The 4th dimension of filter_1 is the number of units, which decides the 3rd dimension of its output
#filter_1_units <- 2
#filter_1 <- array(round(runif(18, min=-3, max=3)), c(3,3,input_depth,filter_1_units))

#The 3rd dimension of filter_2 should be equal to the 4th dimension of filter_1
#The 4th dimension of filter_2 is the number of units, which decides the 3rd dimension of its output
#filter_2_units <- 2
#filter_2 <- array(round(runif(18, min=-3, max=3)), c(3,3,filter_1_units,filter_2_units))

#input * filter_1 + bias_vector_1 = output of deep_1
#bias_vector_1 <- rep(0, dim(filter_1)[4])

#output of deep_1 * filter_2 + bias_vector_2 = output of deep_2
#bias_vector_2 <- rep(0, dim(filter_2)[4])

#final output is the distribution over 10 digits
#num_of_categories <- 10
#bias_vector_3 <- rep(0, num_of_categories)

#If deep_2's max pooling is FALSE:
#step 1: dim(input)[1]-2 = output of convolution_layer
#step 2: output of convolution_layer/2 = output of max pooling
#step 3: output of max pooling-2 = output of convolution_layer
#step 4: the number of all the weights after the deep_2 = width * height * filter_2_units
#height = ceiling((dim(input)[1]-2)/2 - 2)
#width = ceiling((dim(input)[2]-2)/2 - 2)
#weight_matrix_row_count = ceiling((dim(input)[1]-2)/2 - 2) * ceiling((dim(input)[2]-2)/2 - 2) * filter_2_units

#If deep_2's max pooling is TRUE:
#there is one more max pooling layer after step 3: output of convolution_layer/2 = final result
#weight_matrix_row_count = ceiling(((dim(input)[1]-2)/2 - 2)/2) * ceiling(((dim(input)[2]-2)/2 - 2)/2) * filter_2_units

#nrow = the number of all the weights after the deep_2 and flatten_1 = width * height * filter_2_units
#weight_matrix <- matrix(runif(weight_matrix_row_count * num_of_categories), nrow=weight_matrix_row_count, ncol=num_of_categories)

#output <- forward_propagation(input, filter_1, filter_2, weight_matrix, bias_vector_1, bias_vector_2, bias_vector_3, units)
