

EncoderDecoder <-
  R6::R6Class("EncoderDecoder",

    inherit = keras::KerasLayer,

    public = list(
      mode = NULL,
      num_layers = NULL,
      hidden_dims = NULL,
      original_dim = NULL,
      code_dim = NULL,
      activation = NULL,
      hidden_layers = NULL,
      regularizer = NULL,
      batchnorm = NULL,

      initialize = function(mode,
                            num_layers,
                            hidden_dims,
                            original_dim,
                            code_dim,
                            activation,
                            regularizer,
                            batchnorm) {
        self$mode         <- mode
        self$num_layers   <- num_layers
        self$hidden_dims  <- hidden_dims
        self$original_dim <- original_dim
        self$code_dim     <- code_dim
        self$activation   <- activation
        self$regularizer  <- regularizer
        self$batchnorm    <- batchnorm
      },

      build = function(input_shape) {

        if(rlang::is_empty(self$original_dim) & self$mode == "decoder")
          stop("Original dimension must be supplied if mode == \"decoder\".")

        if(self$mode == "decoder" & rlang::is_empty(self$code_dim))
          stop("Code dim must be supplied when mode == \"decoder\".")

        if(rlang::is_empty(self$original_dim))
          self$original_dim <- input_shape[[length(input_shape)]]

        if (self$mode == "decoder")
          self$hidden_dims  <- c(self$hidden_dims, self$original_dim)

        input_dims <- dplyr::lag(self$hidden_dims)

        input_dims[[1]] <-
          if (self$mode == "encoder") self$original_dim else self$code_dim

        get_layer <- function(in_features, out_features) {

          W <- self$add_weight(
            name = "W_",
            shape = list(in_features, out_features),
            initializer = initializer_he_normal(),
            regularizer = self$regularizer,
            trainable = TRUE)

          b <- self$add_weight(
            name = "b_",
            shape = list(out_features),
            initializer = initializer_zeros(),
            regularizer = self$regularizer,
            trainable = TRUE)

          c(W, b)
        }

        self$hidden_layers <-
          purrr::map2(input_dims, self$hidden_dims, get_layer)

      },

      call = function(x, mask = NULL) {

        out <- x

        for (layer in self$hidden_layers) {
          W   <- layer[[1]]
          b   <- layer[[2]]

          out <- tf$add(tf$matmul(out, W), b)

          if (self$batchnorm)
            out <- layer_batch_normalization(out, renorm = TRUE)

          out <- self$activation(out)
        }

        out
      },

      compute_output_shape = function(input_shape) {

        output_dim <- self$hidden_dims[[length(hidden_dims)]]

        list(input_shape[[1]], output_dim)
      }

    )
  )


#' @export
layer_encoder_decoder <-
  function(object,
           mode = 'encoder',
           num_layers = NULL,
           hidden_dims = NULL,
           original_dim = NULL,
           code_dim = NULL,
           activation = 'relu',
           regularizer = NULL,
           batchnorm = FALSE,
           name = NULL,
           trainable = TRUE) {

    create_layer(EncoderDecoder,
                 object,
                 list(
                   mode = tolower(mode),
                   num_layers = as.integer(num_layers),
                   hidden_dims = as.integer(hidden_dims),
                   original_dim = as.integer(original_dim),
                   code_dim = as.integer(code_dim),
                   activation = tf$keras$activations$get(activation),
                   regularizer = tf$keras$regularizers$get(regularizer),
                   batchnorm = as.logical(batchnorm),
                   name = name,
                   trainable = trainable
                 ))
  }



#' Autoencoder keras model wrapper over simple dense layers
#'
#' @param original_dim integer denoting original dimensionality of input space
#' @param hidden_dims may be an integer, or a list containing desired units for
#' encoder step.  `list(512, 256, 64)` will give a `code_dimension` of `64`
#' @param num_layers integer, number of layers to use in each encoder decoder.
#'  [num_layers] must equal [length(hidden_dims)] unless hidden_dims is a scalar,
#'  in which case, hidden_dims is automatically computed based on nearest powers of 2
#' @param regularizer callable or string name of keras regularizer
#' @param use_batchnorm bool indicating use of batchnorm layer inbetween each dense
#' @param name (optional) string to name the model
#' @export
autoencoder_model <-
  function(original_dim,
           hidden_dims,
           num_layers = 3,
           regularizer = NULL,
           use_batchnorm = FALSE,
           name = NULL) {
    # library(tensorflow)
    # library(keras)

    num_layers   <- as.integer(num_layers)
    original_dim <- as.integer(original_dim)

    if (identical(length(hidden_dims), 1L)) {
      hidden_dims <- rev(pow2_range(hidden_dims, original_dim))

      idx <- which(hidden_dims == original_dim)
      if (!rlang::is_empty(idx))
        hidden_dims <- hidden_dims[-idx]

      if (length(hidden_dims) != num_layers)
        num_layers <- length(hidden_dims)
    }

    keras_model_custom(name = name, function(self) {
      self$encoder_layer <-
        layer_encoder_decoder(mode = "encoder",
                              num_layers = num_layers,
                              hidden_dims = hidden_dims,
                              batchnorm = use_batchnorm)

      hidden_dims <- rev(hidden_dims)
      code_dim    <- hidden_dims[[1]]

      self$decoder_layer <-
        layer_encoder_decoder(mode = "decoder",
                              num_layers = num_layers,
                              hidden_dims = hidden_dims,
                              original_dim = original_dim,
                              code_dim = code_dim,
                              batchnorm = use_batchnorm)

      # Call
      function(x, mask = NULL, training = FALSE) {
        x <- self$encoder_layer(x)
        x <- self$decoder_layer(x)

        x
      }
    })
  }













