#assumption: input must be of the shape (x, y, z)
#if input is 2d, it can be transformed as (x, y, 1)
flatten <- function(stacks)
{
  height <- dim(stacks)[1]
  width <- dim(stacks)[2]
  depth <- dim(stacks)[3]
  
  #output should be 1-dimensional vector
  output <- rep(0, height*width*depth)
  
  #position in the output
  x <- 1
  
  for(i in seq(depth))
  {
    for(j in seq(height))
    {
      for(k in seq(width))
      {
        output[x] <- stacks[j, k, i]
        x <- x + 1
      }
    }
  }    
  
  #return the output
  output
}

#testing
#stacks <- array(round(runif(24,min=0,max=10)), c(2,3,4))
#output <- flatten(stacks)