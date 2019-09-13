convolutionOperation <- function(filters, input) 
{
  #pre-process the input to ensure it is of the 2D shape (x, x, 1)
  if(dim(input)[3] != 1)
    stop("only 2d input is allowed!")
  
  inputshape <- dim(input)
  filtershape <- dim(filters)
  
  #steps for filters to take to move along x-axis, and x_steps = ncol(fmaps)
  x_steps = inputshape[2] - filtershape[2] + 1
  
  #teps for filters to take to move along y-axis, and y_steps = nrow(fmaps)
  y_steps = inputshape[1] - filtershape[1] + 1
  
  #steps for filters to take to move along z-axis, a.k.a. depth. 
  #E.g., black-and-white images have 1 depth, and color images have 3 depths
  depth = dim(filters)[3]
  
  
  #initialise stack of feature maps
  fmaps <- array(0, c(x_steps, y_steps, depth))
  
  #initialise the position in fmaps, as to where to update its value
  x <- 1
  y <- 1
  
  #crop increment of input:
  #E.g., if a 3x3 matrix is the cropped matrix from a 9x9 input matrix, then the increment = 3-1 = 2
  increment <- filtershape[1] - 1
  
  #if output matrix = cropped input matrix * filter,
  #Then pos is the position in the output matrix to get its value and pass it to fmaps
  pos <- (filtershape[1] + 1) / 2
  
  #for all filters
  for(i in seq(depth))
  {
    for(j in seq(x_steps))
    {
      for(k in seq(y_steps))
      {
        output <- input[k:k+increment, j:j+increment,] * filters[,,i]
        output <- array_reshape(output, c(filtershape[1], filtershape[2], 1))
        fmaps[x,y,i] <- output[pos, pos, 1]
        y <- y + 1
      }
      
      x <- x + 1
      y <- 1 
    }
    
    x <- 1
    y <- 1
  }
  
  #export feature maps
  fmaps
}

#testing
example_image <- array(runif(100), c(6,6,1))
example_filters <- array(runif(1),c(3,3,2))
featuremaps <- convolutionOperation(example_filters, example_image)

