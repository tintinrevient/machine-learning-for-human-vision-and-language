library(keras)

#change it to your model path
model <- load_model_hdf5("../model/cats_and_dogs_small.h5")

#change it to your image path
img_path <- "../pix/cat.1692.jpg"
img <- image_load(img_path, target_size = c(150, 150))
img_tensor <- image_to_array(img)
img_tensor <- array_reshape(img_tensor, c(1, 150, 150, 3))
img_tensor <- img_tensor / 255

cat("image dimension is:", dim(img_tensor))
cat("plot the original image on the 'Plots' panel.")
plot(as.raster(img_tensor[1,,,]))

#extract the outputs from layer 1 to layer 8
layer_outputs <- lapply(model$layers[1:8], function(layer) layer$output)

#define the model with multiple outputs
activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)

#get the outputs from layer 1 to layer 8 by the single input: img_tensor
activations <- activation_model %>% predict(img_tensor)

#plot a single feature image from one layer's output by one channel
plot_channel <- function(channel) {
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(channel), axes = FALSE, asp = 1, col = terrain.colors(12))
}

#get the output from layer 1 
first_layer_activation <- activations[[1]]

#plot the feature image from the layer 1's output by the second channel (there are 32 channels in total)
# channels = filters
plot_channel(first_layer_activation[1,,,2])

