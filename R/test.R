


keras_model_test <- function(num_classes,
                             use_bn = FALSE, use_dp = FALSE,
                             name = NULL) {

  # define and return a custom model
  keras_model_custom(name = name, function(self) {

    # create layers we'll need for the call (this code executes once)
    self$dense1 <- layer_dense(units = 32, activation = "relu")
    self$dense2 <- layer_dense(units = num_classes, activation = "softmax")
    if (use_dp)
      self$dp <- layer_dropout(rate = 0.5)
    if (use_bn)
      self$bn <- layer_batch_normalization(axis = -1)

    # implement call (this code executes during training & inference)
    function(inputs, mask = NULL) {
      x <- self$dense1(inputs)
      if (use_dp)
        x <- self$dp(x)
      if (use_bn)
        x <- self$bn(x)
      self$dense2(x)
    }
  })
}



keras_model_test2 <- function(num_classes = 10,
                              use_bn = FALSE, use_dp = FALSE,
                              name = NULL) {

  # define and return a custom model
  keras_model_custom(name = name, function(self) {

    # create layers we'll need for the call (this code executes once)
    self$dense1 <- layer_dense(units = 512,
                               kernel_initializer = initializer_he_normal(),
                               activation = "relu")
    self$dense2 <- layer_dense(units = 256,
                               kernel_initializer = initializer_he_normal(),
                               activation = "relu")
    self$dense3 <- layer_dense(units = 64,
                               kernel_initializer = initializer_he_normal(),
                               activation = "relu")
    self$dense4 <- layer_dense(units = as.integer(num_classes),
                               activation = 'softmax')
    if (use_dp)
      self$dp <- layer_dropout(rate = 0.5)
    if (use_bn)
      self$bn <- layer_batch_normalization(axis = -1)

    # implement call (this code executes during training & inference)
    function(x, mask = NULL) {
      x <- self$dense1(x)
      x <- self$dense2(x)
      x <- self$dense3(x)
      if (use_dp)
        x <- self$dp(x)
      if (use_bn)
        x <- self$bn(x)
      self$dense4(x)
    }
  })
}



LayerTest <-
  R6::R6Class("LayerTest",

              inherit = KerasLayer,

              public = list(
                num_classes = NULL,
                dense1 = NULL,
                dense2 = NULL,
                dp = NULL,
                bn = NULL,

                initialize = function(num_classes) {
                  self$num_classes   <- num_classes
                },

                build = function(input_shape) {
                  self$dense1 <- layer_dense(units = 32, activation = "relu")
                  self$dense2 <- layer_dense(units = self$num_classes,
                                             activation = "softmax")
                  if (FALSE)
                    self$dp <- layer_dropout(rate = 0.5)
                  if (FALSE)
                    self$bn <- layer_batch_normalization(axis = -1)
                },

                call = function(x, mask = NULL) {
                  x <- self$dense1(x)
                  if (FALSE)
                    x <- self$dp(x)
                  if (FALSE)
                    x <- self$bn(x)
                  self$dense2(x)
                },

                compute_output_shape = function(input_shape) {
                  output_dim <- list(input_shape[[1]], self$num_classes)
                }

              )
  )

layer_test <-
  function(object,
           num_classes,
           name = NULL,
           trainable = TRUE) {

    create_layer(LayerTest,
                 object,
                 list(
                   num_classes = as.integer(num_classes),
                   name = name,
                   trainable = trainable
                 ))
  }


layer_test_model <- function() {
  input <- layer_input(shape = list(784))

  output <- input %>%
    layer_test(num_classes = 10)

  build_and_compile(input, output)
}


layer_test2_model <- function() {

  input <- layer_input(shape = list(784))

  original_dim <- input$shape[[1]]
  hidden_dims  <- c(512, 256, 64)

  enc_out <- input %>%
    layer_encoder_decoderV2(mode = "encoder",
                            num_layers = 3,
                            hidden_dims = hidden_dims)

  output <- enc_out %>%
    layer_dense(10, activation = 'softmax')

  build_and_compile(input, output)
}




load_mnist <- function() {
  library(keras)

  # Data Preparation --------------------------------------------------------

  batch_size <- 128
  num_classes <- 10
  epochs <- 40

  # The data, shuffled and split between train and test sets
  mnist <- dataset_mnist()
  x_train <- mnist$train$x
  y_train <- mnist$train$y
  x_test <- mnist$test$x
  y_test <- mnist$test$y

  # Redimension
  x_train <- array_reshape(x_train, c(nrow(x_train), 784))
  x_test <- array_reshape(x_test, c(nrow(x_test), 784))

  # Transform RGB values into [0,1] range
  x_train <- x_train / 255
  x_test <- x_test / 255

  cat(nrow(x_train), 'train samples\n')
  cat(nrow(x_test), 'test samples\n')

  # Convert class vectors to binary class matrices
  y_train <- to_categorical(y_train, num_classes)
  y_test  <- to_categorical(y_test, num_classes)

  list(
    x_train = x_train,
    y_train = y_train,
    x_test = x_test,
    y_test = y_test,
    batch_size = batch_size,
    epochs = epochs,
    num_classes = num_classes
  )
}



#' @importFrom zeallot %<-%
test_autoencoder <- function() {
    source("R/utils.R")
    source("R/autoencoder.R")

    library(zeallot)
    library(keras)

    c(x_train, y_train, x_test, y_test,
      batch_size, epochs, num_classes) %<-% load_mnist()

    (model <- autoencoder_model(hidden_dims = 64))

    model %>% compile(
      loss = 'mse',
      optimizer = 'adagrad'
    )

    model %>% fit(x_train,
                  x_train,
                  epochs = epochs,
                  steps_per_epoch = 100,
                  batch_size = batch_size,
                  validation_data = list(x_test, x_test))

   }

