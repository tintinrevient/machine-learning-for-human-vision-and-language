denseLayer <- function(input, units)
{
  #initialise the output with the length of units
  output <- rep(0, units)
  
  #initialise the weight matrix (which should NOT be hard-coded, and it shoud be learned during the training)
  number_of_weights <- length(input) * units
  weight_matrix <- matrix(runif(number_of_weights), nrow=length(input), ncol=units)
  
  #initialise the bias (which should NOT be hard-coded, and it should be learned during the training)
  bias_vector <- c(runif(units))
  
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
output <- denseLayer(input, units)

