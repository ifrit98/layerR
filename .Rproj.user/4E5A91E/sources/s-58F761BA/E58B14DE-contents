
conv_1d <- function(x, filters, kernel_size, ..., separable = TRUE,
                   use_bias = FALSE, activation = 'relu', padding = 'same') {
  conv_layer <- if (separable)
    layer_separable_conv_1d
  else
    layer_conv_1d

  conv_layer(x, filters, kernel_size,
             use_bias = use_bias,
             activation = activation,
             padding = padding, ...)
}



is_scalar <- function(x) identical(length(x), 1L)

#' @export
conv_block_1d <- function(input,
                          filter_sizes,
                          kernel_sizes,
                          depth = if (is_scalar(filter_sizes)) 2 else length(filter_sizes),
                          separable = TRUE,
                          cap = layer_max_pooling_1d,
                          return_all = TRUE, ...) {
  stopifnot(is_scalar_integerish(depth),
            is_integerish(filter_sizes),
            is_integerish(kernel_sizes))
  filter_sizes %<>% rep(depth)
  kernel_sizes %<>% rep(depth)

  if (is.character(cap)) {
    cap <- if (!nzchar(cap))
      NULL
    else
      switch(
        cap,
        max = layer_max_pooling_1d,
        mean = ,
        ave = ,
        average = layer_average_pooling_1d,
        batchnorm = layer_batch_normalization,
        none = ,
        "NULL" = NULL,
        stop("Don't recognize pooling operation %s", cap)
      )
  }

  if (!is.null(cap))
    cap <- rlang::as_function(cap)

  top <- input
  stack <- vector("list", depth + length(cap))

  for (i in seq_len(depth)) {
    stack[[i]] <- top <-
      conv_1d(top, filter_sizes[i], kernel_sizes[i], separable = separable, ...)
  }

  if (!is.null(cap))
    stack[[length(stack)]] <- top <- cap(top)

  if (return_all)
    stack
  else
    top
}



#' @export
conv_tower <- function(input,
                       filter_sizes = c(32, 64, 128, 256),
                       blocks = if (is_scalar(filter_sizes)) 4 else length(filter_sizes),
                       block_heights = 3, # depth of conv_blocK_1d
                       kernel_sizes = 8,
                       block_caps = layer_max_pooling_1d,
                       last_cap = if (length(block_caps) == blocks)
                         last(block_caps)
                       else if (grepl("all", return))
                         layer_max_pooling_1d
                       else
                         layer_global_max_pooling_1d,
                       separable = TRUE,
                       return = "top",
                       ...) {
  stopifnot(is_scalar_integerish(blocks), blocks >= 1L)
  force(last_cap)
  force(block_heights)

  if (is_scalar(block_heights))
    block_heights %<>% rep(blocks)
  if (is_scalar(kernel_sizes))
    kernel_sizes  %<>% rep(blocks)
  if (is_scalar(filter_sizes))
    filter_sizes %<>% rep(blocks)

  if (length(block_caps) == 1) {
    block_caps <- lapply(1:blocks, function(x) block_caps)
    if (!is.list(last_cap))
      last_cap %<>% list()
    block_caps[ blocks + 1 - rev(seq_along(last_cap)) ] <- last_cap
  } else
    stopifnot(length(block_caps) == blocks)

  stopifnot(return %in% c("top", "block_caps", "all_conv_layers", "all"))

  return_all <- return %in% c("all", "all_conv_layers")

  top   <- input
  tower <- vector("list", blocks)

  for (i in seq_len(blocks)) {
    tower[[i]] <- top <-
      conv_block_1d(
        top,
        filter_sizes = filter_sizes[[i]],
        kernel_sizes = kernel_sizes[[i]],
        depth = block_heights[[i]],
        cap = block_caps[[i]],
        separable = separable,
        return_all = return_all,
        ...
      )
  }

  switch(return,
         top = top,
         block_tops = tower,
         all_conv_layers = ,
         all = unlist(tower, recursive = FALSE))
}


