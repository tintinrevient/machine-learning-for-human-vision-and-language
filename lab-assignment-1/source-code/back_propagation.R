library(keras)

source("~/Documents/workspace/machine-learning-for-human-vision-and-language/lab-assignment-1/source-code/convolution_layer.R")
source("~/Documents/workspace/machine-learning-for-human-vision-and-language/lab-assignment-1/source-code/forward_propagation.R")


loss <- function(output, target)
{
  sum((target - output)^2) / length(target)
}

layer_activation <- function(input, filter_1, filter_2, weight_matrix, bias_vector_1, bias_vector_2, bias_vector_3, units)
{
  deep_1 <- deep_layer(input, filter_1, bias_vector_1, TRUE)
  deep_2 <- deep_layer(deep_1, filter_2, bias_vector_2, TRUE)
  
  flatten_1 <- flatten(deep_2)
  dense_1 <- dense_layer(flatten_1, units, weight_matrix, bias_vector_3)
  softmax_1 <- softmax(dense_1)
  
  #output
  list(deep_1, deep_2, softmax_1)
}


output_derivatives <- function(input, filter_1, filter_2, weight_matrix, bias_vector_1, bias_vector_2, bias_vector_3, units, target, layeractivation)
{
  #calculate output
  deep_2 <- layeractivation[2]
  
  flatten_1 <- flatten(deep_2)
  dense_1 <- dense_layer(flatten_1, units, weight_matrix, bias_vector_3)
  output <- softmax(dense_1)
  
  #make empty array for derivatives per node
  derivatives <- array(data=0, dim=dim(output))
  
  #calculate derivative of Loss by activation per output node
  for(x in seq(length(output)))
  {
    activation <- layeractivation[3][x]
    
    lossfunction <- function(activation)
    {
      new_output <- output
      new_output[x] <- activation
      loss(new_output, target)
    }
    derivatives[x] <- grad(lossfunction, activation)

  }
  
  derivatives
  
}

layer_2_derivatives <- function(input, filter_1, filter_2, weight_matrix, bias_vector_1, bias_vector_2, bias_vector_3, units, target, layeractivation, output_derivatives)
{
  #calculate output
  deep_2 <- layeractivation[2]
  
  #make empty array for derivatives per node
  derivatives <- array(data=0, dim=dim(deep_2))
  
  #calculate loss gradient by activation per output node
  for(x in seq(dim(deep_2)[1]))
  {
    for(y in seq(dim(deep_2)[2]))
    {
      for(z in seq(dim(deep_2)[3]))
      {
        activation <- layer_activation[2][x,y,z]
        
        gradient <- function(activation)
        {
          new_deep_2 <- deep_2
          new_deep_2[x,y,z] <- activation
          
          flatten_1 <- flatten(deep_2)
          dense_1 <- dense_layer(flatten_1, units, weight_matrix, bias_vector_3)
          new_output <- softmax(dense_1)
          old_output <- layer_activation[3]
          delta <- new_output - old_output
          delta %*% output_derivatives / length(delta)
          
        }
        derivatives[x,y,z] <- gradient(activation)
      }
    }
    
    
  }
  
  derivatives
}




back_propagation <- function(input, filter_1, filter_2, weight_matrix, bias_vector_1, bias_vector_2, bias_vector_3, units, target, r)
{
  print("training weightmatrix")
  new_weight_matrix <- weight_matrix
  
  for(y in seq(dim(weight_matrix)[2]))
  {
    for(x in seq(dim(weight_matrix)[1]))
    {
      weight <- weight_matrix[x,y]
    
      
      lossfunction <- function(w) 
      {
        newmatrix <- weight_matrix
        newmatrix[x,y] <- w
        
        output <- forward_propagation(input, filter_1, filter_2, newmatrix, bias_vector_1, bias_vector_2, bias_vector_3, units)
        loss <- sum((target - output)^2) / length(target)
        loss
      }
      
      weight_derivative <- grad(lossfunction, weight)
      new_weight <- weight - r * weight_derivative
      
      new_weight_matrix[x,y] <- new_weight
    }  
  }
  
  print("training filter 1")
  new_filter_1 <- filter_1
  
  for(v in seq(dim(filter_1)[1])) 
  {
    for(x in seq(dim(filter_1)[2]))
    {
      for(y in seq(dim(filter_1)[3]))
      {
        for(z in seq(dim(filter_1)[4]))
        {
          weight <- filter_1[v,x,y,z]
          
          
          lossfunction <- function(w) 
          {
            newfilter <- filter_1
            newfilter[v,x,y,z] <- w
            output <- forward_propagation(input, newfilter, filter_2, weight_matrix, bias_vector_1, bias_vector_2, bias_vector_3, units)
            loss <- sum((target - output)^2) / length(target)
            loss
          }
          
          
          weight_derivative <- grad(lossfunction, weight)
          new_weight <- weight - r * weight_derivative
          
          new_filter_1[v,x,y,z] <- new_weight
        }
      }
    }  
  }
  
  print("training filter 2")
  
  new_filter_2 <- filter_2
  
  for(v in seq(dim(filter_2)[1]))
  {
    for(x in seq(dim(filter_2)[2]))
    {
      for(y in seq(dim(filter_2)[3]))
      {
        for(z in seq(dim(filter_2)[4]))
        {
          weight <- filter_2[v,x,y,z]
          
          lossfunction <- function(w) 
          {
            newfilter <- filter_2
            newfilter[v,x,y,z] <- w
            output <- forward_propagation(input, filter_1, newfilter, weight_matrix, bias_vector_1, bias_vector_2, bias_vector_3, units)
            loss <- sum((target - output)^2) / length(target)
            loss
          }
          
          
          weight_derivative <- grad(lossfunction, weight)
          new_weight <- weight - r * weight_derivative
          
          new_filter_2[v,x,y,z] <- new_weight
        }
      }
    }  
  }
  
  print("training bias 1")
  
  new_bias_1 <- bias_vector_1
  
  for (x in seq(length(bias_vector_1)))
  {
    weight <- bias_vector_1[x]
    
    lossfunction <- function(w) 
    {
      newvector <- bias_vector_1
      newvector[x] <- w
      output <- forward_propagation(input, filter_1, filter_2, weight_matrix, newvector, bias_vector_2, bias_vector_3, units)
      loss <- sum((target - output)^2) / length(target)
      loss
    }
    
    
    weight_derivative <- grad(lossfunction, weight)
    new_weight <- weight - r * weight_derivative
    new_bias_1[x] <- new_weight
  }
  
  print("training bias 2")
  
  new_bias_2 <- bias_vector_2
  
  for (x in seq(length(bias_vector_2)))
  {
    weight <- bias_vector_2[x]
    
    lossfunction <- function(w) 
    {
      newvector <- bias_vector_2
      newvector[x] <- w
      output <- forward_propagation(input, filter_1, filter_2, weight_matrix, bias_vector_1, newvector, bias_vector_3, units)
      loss <- sum((target - output)^2) / length(target)
      loss
    }
    
    
    weight_derivative <- grad(lossfunction, weight)
    new_weight <- weight - r * weight_derivative
    new_bias_2[x] <- new_weight
  }
  
  print("training bias 3")
  
  new_bias_3 <- bias_vector_3
  
  for (x in seq(length(bias_vector_3)))
  {
    weight <- bias_vector_3[x]
    
    lossfunction <- function(w) 
    {
      newvector <- bias_vector_3
      newvector[x] <- w
      output <- forward_propagation(input, filter_1, filter_2, weight_matrix, bias_vector_1, bias_vector_2, newvector, units)
      loss <- sum((target - output)^2) / length(target)
      loss
    }
    
    
    weight_derivative <- grad(lossfunction, weight)
    new_weight <- weight - r * weight_derivative
    new_bias_3[x] <- new_weight
  }
  
  #return new filters and weights
  list(new_filter_1, new_filter_2, new_weight_matrix, new_bias_1, new_bias_2, new_bias_3)
  
}

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

#testing
input <- x_train[1,,,]
input <- array_reshape(mnist_image, c(28, 28, 1))
target <- y_train[1,]


filter_1 <- array(round(runif(90, min=-3, max=3)), c(3,3,1,10))
filter_2 <- array(round(runif(180, min=-3, max=3)), c(3,3,10,10))
units <- 10
bias_vector_1 <- rep(0, dim(filter_1)[4])
bias_vector_2 <- rep(0, dim(filter_2)[4])
bias_vector_3 <- rep(0, units)

count <- 360
weight_matrix <- matrix(runif(count * units), nrow=count, ncol=units)
r <- 0.01

output <- back_propagation(input, filter_1, filter_2, weight_matrix, bias_vector_1, bias_vector_2, bias_vector_3, units, target, r)


layeractivation <- layer_activation(input, filter_1, filter_2, weight_matrix, bias_vector_1, bias_vector_2, bias_vector_3, units)
outputderivatives <- output_derivatives(input, filter_1, filter_2, weight_matrix, bias_vector_1, bias_vector_2, bias_vector_3, units, target, layeractivation)
layer2derivatives <- layer_2_derivatives(input, filter_1, filter_2, weight_matrix, bias_vector_1, bias_vector_2, bias_vector_3, units, target, layeractivation, output_derivatives)
layer1derivatives <- layer_1_derivatives(input, filter_1, filter_2, weight_matrix, bias_vector_1, bias_vector_2, bias_vector_3, units, target, layeractivation, layer2derivatives)