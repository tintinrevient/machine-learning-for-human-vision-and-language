#assumption: input must be of the shape (x, y, z)
#if input is 2d, it can be transformed as (x, y, 1)

#question: normalization layer is performed stack by stack or on all stacks?
normalisation <- function(stacks)
{
  for(i in seq(dim(stacks)[3]))
  {
    #input
    input <- stacks[,,i]
    
    #mean
    mean <- mean(input)
    
    #substract mean from input
    tmp <- input - mean
    
    #standard deviation
    sd <- sqrt(sum(tmp^2)/(length(input)-1))
    
    #if sd = 0, just skip
    if(sd == 0)
      stacks[,,i] <- tmp
    else
      stacks[,,i] <- tmp/sd
  }
  
  #return the output
  stacks
}

#testing
stacks <- array(c(10,8,-6,-3,10,8,-6,-3), dim=c(2,2,2))
output <- normalisation(stacks)

#verify if the mean is 0
mean <- mean(output[,,1])
mean <- mean(output[,,2])

#verify if the standard deviation is 1
sd <- sd(output[,,1])
sd <- sd(output[,,2])