normalisation <- function(input)
{
  #get mean
  length = length(input)
  sum = sum(input)
  mean = sum/length
  
  #substract mean from input
  tmp = input - mean(input)
  
  #calculate standard deviation
  sd = sqrt(sum(tmp^2)/(length-1))
  
  #divide by sd, don't do anything if sd = 0
  if(sd == 0)
    output = tmp
  else
    output = tmp/sd
  
  #return the output
  output
}

#testing
input <- matrix(c(10,8,-6,-3), nrow=2, ncol=2)
output <- normalisation(input)

#verify if the mean is 0
mean <- mean(output)

#verify if the standard deviation is 1
sd <- sqrt(sum((output - mean(output))^2) / (length(output) - 1))
