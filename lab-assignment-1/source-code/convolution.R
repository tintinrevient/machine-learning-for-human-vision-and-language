convolutionOperation <- function(filters, input) 
{
  
  inputshape <- dim(input)
  filtershape <- dim(filters)
  
  #steps for filters to take to move along x-axis, so x_steps = ncol(fmaps)
  x_steps <- inputshape[2] - filtershape[2] + 1
  
  #steps for filters to take to move along y-axis, so y_steps = nrow(fmaps)
  y_steps <- inputshape[1] - filtershape[1] + 1
  
  #steps for filters to take to move along z-axis, a.k.a. depth. 
  #E.g., black-and-white images have depth 1, and color images have depth 3
  depth <- filtershape[4]
  
  #initialise stack of feature maps
  fmaps <- array(0, c(y_steps, x_steps, depth))
  
  #size of the interval in the input image that the filter will cover
  increment <- filtershape[1] - 1
  
  #loop through filters
  for(i in seq(depth))
  {
    #loop through input image
    for(j in seq(x_steps))
    {
      for(k in seq(y_steps))
      {
        #elementwise multiplication of filter with current section of input image
        output <- input[j:(j+increment), k:(k+increment),] * filters[,,,i]
        #sum the output and store it in the featuremaps
        fmaps[j,k,i] <- sum(output)
      }
    }
  }
  
  #export feature maps
  fmaps
}

#testing
example_image <- array(round(runif(1875, min=-3, max=3)), c(25, 25, 3))
example_filters <- array(round(runif(18, min=-3, max=3)),c(5, 5, 3, 2))
output <- convolutionOperation(example_filters, example_image)
