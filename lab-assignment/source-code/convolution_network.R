library(keras)

# import the dataset
mnist <- dataset_mnist()
c(c(x_train, y_train), c(x_test, y_test)) %<-% mnist

# reshape from 3D to 2D
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))

# rescale from the interval [0,255] to [0,1]
x_train <- x_train / 255
x_test <- x_test / 255

# rescale from the interval [0, 9] to [0, 1]
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# model with two layers
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = 'relu', input_shape = c(28,28,1)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  # layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  # layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

history <- model %>% fit(
  x_train, y_train,
  batch_size = 128,
  epochs = 6,
  verbose = 1,
  validation_split = 0.2
)

score <- model %>% evaluate(
  x_test, y_test,
  verbose = 0
)