relu <- function(input)
{
  shape <- dim(input)

  for(z in seq(shape[3])){
    
    for (x in seq(shape[1]))
    {
      for (y in seq(shape[2]))
        input[x, y, z] <- max(input[x, y, z], 0)
    }
  }
  
  #return the updated feature map
  input
}

#test
input <- array(round(runif(18,min=-10,max=10)), dim=c(3, 3, 2))
output <- relu(input)
