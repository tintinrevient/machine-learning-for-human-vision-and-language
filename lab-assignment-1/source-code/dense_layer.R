#assumption: input must be a sequence, a.k.a, 1-dimensional vector
#assumption: bias_vector's length should be equal to units
dense_layer <- function(input, units, weight_matrix, bias_vector)
{
  #initialise the output with the length of units
  output <- rep(0, units)
  
  #fill in the value for each output unit
  for(i in seq(units))
  {
    output[i] <- input %*% weight_matrix[,i] + bias_vector[i]
  }
  
  #return the output
  output
}

#testing
input <- c(round(runif(10, min=1, max=10)))
units <- 5
weight_matrix <- matrix(runif(length(input) * units), nrow=length(input), ncol=units)
bias_vector <- rep(0, units)

output <- dense_layer(input, units, weight_matrix, bias_vector)

