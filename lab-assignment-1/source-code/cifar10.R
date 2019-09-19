library(keras)

cifar10 <- dataset_cifar10()
c(c(x_train, y_train), c(x_test, y_test)) %<-% cifar10

x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = 'relu', input_shape = c(32, 32, 3), padding = "same") %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = 'relu', padding = "same") %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>%
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = c('accuracy')
)

history <- model %>% fit(
  x_train, y_train,
  batch_size = 32,
  epochs = 20,
  verbose = 1,
  validation_data = list(x_test, y_test),
  shuffle = TRUE
)

