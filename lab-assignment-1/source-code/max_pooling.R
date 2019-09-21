#assumption: stacks must be of the shape (x, y, z)
#if stacks is 2d, it can be transformed as (x, y, 1)
max_pooling <- function(stacks, rows, cols)
{
  depth = dim(stacks)[3]
  width <- dim(stacks)[2]
  height <- dim(stacks)[1]
  
  #strides which the max pooling takes to move along x-axis and y-axis
  stride_x <- rows
  stride_y <- cols
  
  #steps which the max pooling takes to move along x-axis, and x_steps = ncol(output)
  x_steps <- ceiling(width / cols)
  
  #steps which the max pooling takes to move along y-axis, and y_steps = nrow(output)
  y_steps <- ceiling(height / rows)
  
  #output returned by the max pooling function
  output <- array(0, dim=c(y_steps, x_steps, depth))
  
  for(z in seq(depth))
  {
    #initial position in the output matrix, which is (0, 0)
    x <- 1
    y <- 1
    
    for(j in seq(y_steps))
    {
      for(i in seq(x_steps))
      {
        output[j,i,z] <- max(stacks[x:min((x+stride_x-1),height), y:min((y+stride_y-1), width), z])
        y <- y + stride_y
      }
      y <- 1
      x <- x + stride_x
    }
  }
  
  #return the stacks
  output
}

#testing
stacks <- array(round(runif(70, min=0, max=10)), dim=c(7,5,2))
output <- max_pooling(stacks, 3, 3)
