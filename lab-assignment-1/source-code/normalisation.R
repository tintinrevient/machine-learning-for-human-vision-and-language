normalisation <- function(input)
{
  length = length(input)
  sum = sum(input)
  mean = sum/length
  
  #tmp = input - mean(input)
  tmp = input - mean(input)
  
  #sd = std(tmp)
  sd = sqrt(sum(tmp^2)/(length-1))
  
  #output = tmp/std(tmp)
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