#assumption: input must be a sequence, a.k.a, 1-dimensional vector
softmax <- function(input)
{
  #length of input
  length <- length(input)

  #initialise the output
  output <- rep(0, length)
  
  #natural log base
  e = exp(1)
  
  sum <- sum(e^input)
  output <- (e^input)/sum
  
  #return the output
  output
}

#testing
input <- c(2,1,0.1)
output <- softmax(input)
sum <- sum(output)