convolutionOperation <- function(filters, input) 
{
  inputshape <- dim(input)
  filtershape <- dim(filters)
  #intialise stack of feature maps
  fmaps <- matrix(0, c(inputshape[1] -2, inputshape[2] -2, filtershape[3]))
  
  #for all filters
  for(j in seq(filtershape[3]))
  {
    #loop through image coordinates
    for(x in seq(inputshape[1] - 2))
    {
      for(y in seq(inputshape[2] - 2))
      {
        #apply filter to each position
        image_sample <- input[x : x + filtershape[1] , y : y + filtershape [2], ]
        filter <- filters[,,j]
        feature <- image_sample * filter
        #save result in feature map
        fmap[x,y,j] <- feature

      }
    }

  }
  
  #export feature maps
  fmaps
}

#testing
mnist <- dataset_mnist()
x <- array_reshape(mnist$test$x, c(10000, 28, 28, 1))
x <- x / 255

example_image <- x[1,,,]
example_image <- array_reshape(example_image, c(28,28,1))
example_filters <- matrix(1, c(3,3,32))

featuremaps <- convolutionOperation(example_filters, example_image)