#assumption: input must be of the shape (x, y, z)
#assumption: bias_vector's length should be equal to units of the filters
convolution_layer <- function(input, filter, bias_vector) 
{
  #convolution only processes one image at a time, so filter and input are only for one image
  #if input is of the shape (x, x, 3), filters should be of the shape (x', x', 3, units)
  #if input is of the shape (x, x, 1), filters should be of the shape (x', x', 1, units)
  #if filters are of the shape (x', x', y, units), bias_vector's length should be equal to units
  #prerequisite: x > x'
  inputshape <- dim(input)
  filtershape <- dim(filter)
  
  #steps for filter to take to move along x-axis, so x_steps = ncol(stacks)
  x_steps <- inputshape[2] - filtershape[2] + 1
  
  #steps for filter to take to move along y-axis, so y_steps = nrow(stacks)
  y_steps <- inputshape[1] - filtershape[1] + 1
  
  units <- dim(filter)[4]
  
  #initialise stack of feature maps
  stacks <- array(0, c(y_steps, x_steps, units))
  
  #increment along x-axis to crop the input image
  x_increment <- filtershape[1] - 1
  
  #increment along y-axis to crop the input image
  y_increment <- filtershape[2] - 1
  
  #loop through units, a.k.a. depth of stacks
  for(k in seq(units)) 
  {
    for(j in seq(y_steps))
    {
      for(i in seq(x_steps))
      {
        #elementwise multiplication of filter with current section of input image
        output <- input[i:(i+x_increment), j:(j+y_increment),] * filter[,,,k] + bias_vector[k]
        #sum the output and store it in the featuremaps
        stacks[i,j,k] <- sum(output)
      }
    }
  }
  
  #export stacks
  stacks
}

#testing
#test_input is of the shape (4, 4, 3)
test_input <- array(round(runif(36, min=-3, max=3)), c(4, 4, 3))

#test_filter is of the shape (3, 3, 3, 2)
#its third dimension 3 is equal to the depth of the test_input
#its fourth dimension 2 is the number of the units of filters
test_filter <- array(round(runif(54, min=-1, max=1)),c(3, 3, 3, 2))

#length of the test_bias_vector is 2, which is equal to test_filter's number of units
test_bias_vector <- rep(0, 2)

output <- convolution_layer(test_input, test_filter, test_bias_vector)
