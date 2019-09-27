#assumption: input must be of the shape (x, y, z)
#if input is 2d, it can be transformed as (x, y, 1)
relu <- function(stacks)
{
  shape <- dim(stacks)

  for(z in seq(shape[3])){
    
    for (y in seq(shape[2]))
    {
      for (x in seq(shape[1]))
        stacks[x, y, z] <- max(stacks[x, y, z], 0)
    }
  }
  
  #return the updated feature map
  stacks
}

#test
#stacks <- array(round(runif(18,min=-10,max=10)), dim=c(3, 3, 2))
#output <- relu(stacks)
