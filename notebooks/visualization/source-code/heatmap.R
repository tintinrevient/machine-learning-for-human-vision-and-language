library(keras)

K <- backend()

model <- application_vgg16(weights = "imagenet")

img_path <- "../pix/creative_commons_elephant.jpg"

img <- image_load(img_path, target_size = c(224, 224)) %>%
  image_to_array() %>%
  array_reshape(dim = c(1, 224, 224, 3)) %>%
  imagenet_preprocess_input() #this does channel-wise color normalization

preds <- model %>% predict(img)

#top-3 predictions
print(imagenet_decode_predictions(preds, top = 3)[[1]])

#get the index of top-1 preds
cat("the index of preds[1,] is: ", which.max(preds[1,]))

african_elephant_output <- model$output[, 387]
last_conv_layer <- model %>% get_layer("block5_conv3")

grads <- K$gradients(african_elephant_output, last_conv_layer$output)[[1]]
pooled_grads <- K$mean(grads, axis = c(0L, 1L, 2L))

iterate <- K$`function`(list(model$input), list(pooled_grads, last_conv_layer$output[1,,,]))
c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(img))

#number of filters is 512
#"how important this channel is" with regard to the elephant class
for (i in 1:512) {
  conv_layer_output_value[,,i] <- conv_layer_output_value[,,i] * pooled_grads_value[[i]]
}

heatmap <- apply(conv_layer_output_value, c(1,2), mean)
heatmap <- pmax(heatmap, 0)
heatmap <- heatmap / max(heatmap)

display_heatmap <- function(heatmap, col = terrain.colors(12)) {
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}

#display the heatmap which contributes to "african_elephant"
#display_heatmap(heatmap)

write_heatmap <- function(heatmap, filename, width = 224, height = 224,
                          bg = "white", col = terrain.colors(12)) {
  
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0,0,0,0))
  on.exit({par(op); dev.off()}, add = TRUE)
  
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}

library(magick)
library(viridis)

image <- image_read(img_path)
info <- image_info(image)
geometry <- sprintf("%dx%d!", info$width, info$height)

pal <- col2rgb(viridis(20), alpha = TRUE)
alpha <- floor(seq(0, 255, length = ncol(pal)))
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)

write_heatmap(heatmap, "elephant_overlay.png",
              width = 14, height = 14, bg = NA, col = pal_col)

image_read("elephant_overlay.png") %>%
  image_resize(geometry, filter = "quadratic") %>%
  image_composite(image, operator = "blend", compose_args = "20") %>%
  plot()