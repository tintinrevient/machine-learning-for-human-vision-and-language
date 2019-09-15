softmax <- function(input)
{
  #length of input
  length <- length(input)

  #initialise the output
  output <- rep(0, length)
  
  #natural log base
  e = exp(1)
  
  sum <- 0
  for(i in seq(length))
    sum <- sum + e^input[i]
  
  output <- (e^input)/sum
  
  #return the output
  output
}

#testing
input <- c(2,1,0.1)
output <- softmax(input)
sum <- sum(output)