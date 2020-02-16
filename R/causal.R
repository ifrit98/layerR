
#' Causal convolution layer, masks out future (look-ahead) sequences
#' @export
layer_causal_conv1d <-
  function(object,
           filters,
           kernel_size,
           strides = 1L,
           dilation_rate = 1L,
           activation = NULL,
           use_bias = TRUE,
           kernel_initializer = tf$keras$initializers$glorot_uniform(),
           bias_initializer = tf$keras$initializers$zeros(),
           kernel_regularizer = NULL,
           bias_regularizer = NULL,
           activity_regularizer = NULL,
           kernel_constraint = NULL,
           bias_constraint = NULL,
           trainable = TRUE,
           name = "causal_conv1d",
           ...) {
    layer_lambda(object, function(x) {

      filters       %<>% as.integer()
      kernel_size   %<>% as.integer()
      strides       %<>% as.integer()
      dilation_rate %<>% as.integer()

      padding <- (kernel_size - 1L) * dilation_rate

      input <- x %>%  tf$pad(tf$constant(list(c(0L, 0L),
                                              c(1L * padding, 0L),
                                              c(0L, 0L))))

      layer_conv_1d(
        input,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        dilation_rate = dilation_rate,
        activation = activation,
        use_bias = use_bias,
        kernel_initializer = kernel_initializer,
        bias_initializer = bias_initializer,
        bias_regularizer = bias_regularizer,
        activity_regularizer = activity_regularizer,
        kernel_constraint = kernel_constraint,
        bias_constraint = bias_constraint,
        trainable = trainable,
        name = name
      )

    }, ...)

  }

