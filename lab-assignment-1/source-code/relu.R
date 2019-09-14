relu <- function(input)
{
  for (x in seq(nrow(input)))
  {
    for (y in seq(ncol(input)))
      input[x, y] <- max(input[x, y], 0)
  }
  
  #return the updated x
  input
}

#test
input <- matrix(round(runif(9,min=-10,max=10)), nrow=3, ncol=3)
output <- relu(input)
