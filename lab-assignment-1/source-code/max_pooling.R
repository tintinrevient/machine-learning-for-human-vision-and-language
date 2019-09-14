maxPooling <- function(input, max_pooling)
{
  #pre-process the input/pooling data to ensure they have the right shape
  if(dim(input)[1] != dim(input)[2])
    stop("Input should be of shape (x, x), where x = x.")
  
  if(dim(max_pooling)[1] != dim(max_pooling)[2])
    stop("Max pooling should be of shape (y, y), where y = y.")
  
  if(dim(input)[1] %% dim(max_pooling)[1] != 0)
    stop("If the shape of input is (x, x) and the shape of max pooling is (y, y), (x % y) shoud be 0.")
  
  #strides which the max pooling takes to move along x-axis and y-axis
  stride <- dim(max_pooling)[1]
  
  #steps which the max pooling takes to move along x-axis
  x_steps <- dim(input)[1] / dim(max_pooling)[1]
  
  #steps which the max pooling takes to move along y-axis
  y_steps <- x_steps
  
  #output returned by the max pooling function
  output <- matrix(0, nrow=y_steps, ncol=x_steps)
  
  #initial position in the output matrix, which is (0, 0)
  x <- 1
  y <- 1
  
  for(i in seq(x_steps))
  {
    for(j in seq(y_steps))
    {
      #matrix position x from input, as to where to crop the input
      pos_x <- 1 + stride*(j-1)
      
      #matrix position y from input, as to where to crop the input
      pos_y <- 1 + stride*(i-1)
      
      output[x,y] <- max(input[pos_x:(pos_x+stride-1), pos_y:(pos_y+stride-1)])
      x <- x + 1
    }
    y <- y + 1
    x <- 1
  }
  
  #return the output
  output
}

#testing
input <- matrix(round(runif(16, min=0, max=10)), nrow=4, ncol=4)
max_pooling <- matrix(0, nrow=2, ncol=2)
output <- maxPooling(input, max_pooling)


