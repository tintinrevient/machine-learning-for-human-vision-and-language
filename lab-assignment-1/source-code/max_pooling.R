maxPooling <- function(input, max_pooling)
{
  #add padding if image size is not a multiple of pooling size
  
  #... for number of rows
  if(dim(input)[1] %% dim(max_pooling)[1] != 0)
  {
    difference <- dim(input)[1] %% dim(max_pooling)[1]
    padding <- matrix(0, ncol=dim(input)[2], nrow=difference)
    input <- rbind(input, padding)
  }
  
  #... for number of columns
  if(dim(input)[2] %% dim(max_pooling)[2] != 0)
  {
    difference <- dim(input)[2] %% dim(max_pooling)[2]
    padding <- matrix(0, nrow=dim(input)[1], ncol=difference)
    input <- cbind(input, padding)
  }

  
  #strides which the max pooling takes to move along x-axis and y-axis
  stride_x <- dim(max_pooling)[1]
  stride_y <- dim(max_pooling)[2]
  
  #steps which the max pooling takes to move along x-axis
  x_steps <- dim(input)[1] / dim(max_pooling)[1]
  
  #steps which the max pooling takes to move along y-axis
  y_steps <- dim(input)[2] / dim(max_pooling)[2]
  
  #output returned by the max pooling function
  output <- matrix(0, nrow=x_steps, ncol=y_steps)
  
  #initial position in the output matrix, which is (0, 0)
  x <- 1
  y <- 1
  
  for(i in seq(x_steps))
  {
    for(j in seq(y_steps))
    {
      #matrix position x from input, as to where to crop the input
      pos_x <- 1 + stride_x*(j-1)
      
      #matrix position y from input, as to where to crop the input
      pos_y <- 1 + stride_y*(i-1)
      

      output[y,x] <- max(input[pos_y:(pos_y+stride_y-1), pos_x:(pos_x+stride_x-1)])
      x <- x + 1
    }
    y <- y + 1
    x <- 1
  }
  
  #return the output
  output
}


#testing
input <- matrix(round(runif(20, min=0, max=10)), nrow=4, ncol=5)
max_pooling <- matrix(0, nrow=2, ncol=2)
output <- maxPooling(input, max_pooling)
