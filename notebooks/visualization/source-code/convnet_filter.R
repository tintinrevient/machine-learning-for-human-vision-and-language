library(keras)
library(grid)

K <- backend()

model <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE
)

#normalize the input to the range [0, 1]
deprocess_image <- function(x) {
  dms <- dim(x)
  x <- x - mean(x)
  x <- x / (sd(x) + 1e-5)
  x <- x * 0.1
  x <- x + 0.5
  x <- pmax(0, pmin(x, 1))
  array(x, dim = dms)
}

generate_pattern <- function(layer_name, filter_index, size = 150) {
  layer_output <- model$get_layer(layer_name)$output
  
  filter_output <- K$mean(layer_output[,,,filter_index])
  grads <- K$gradients(filter_output, model$input)[[1]]
  
  #normalize the gradient tensor by dividing it by its L2 norm
  grads <- grads / (K$sqrt(K$mean(K$square(grads))) + 1e-5)
  
  iterate <- K$`function`(list(model$input), list(filter_output, grads))
  
  input_img_data <- array(runif(size * size * 3), dim = c(1, size, size, 3)) * 20 + 128
  
  step <- 1
  for (i in 1:40) {
    c(filter_output_value, grads_value) %<-% iterate(list(input_img_data))
    input_img_data <- input_img_data + (grads_value * step)
  }
  
  img <- input_img_data[1,,,]
  deprocess_image(img)
}

layer_name <- "block3_conv1"
filter_index <- 1
cat("pattern that channel", filter_index, "in layer", layer_name, "responds to maximally")

grid.raster(generate_pattern(layer_name, filter_index))