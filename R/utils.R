#' Is x explicitly a list type
#' @export
is_list <- function(x) is.vector(x) && !is.atomic(x)

#' Is x explicitly a vector with length(vector) > 1
#' @export
is_vec  <- function(x) is.vector(x) & length(x) != 1L

#' Is x an (array, list, vector) with length(x) > 1
#' @export
is_vec2 <- function(x) is_list(x) | is_vec(x)

#' Are we makring the end or absence of something?
#' @export
is_sentinel <- function(x)
  c(is_empty(x), is.na(x), is.nan(x), is.null(x), is.infinite(x)) %>% any()


#' Convencience fun to one-liner a model with in-outs
#' @export
build_and_compile <-
  function(input,
           output,
           optimizer = 'adam',
           loss = "mse",
           metric = 'acc') {
    model <- keras::keras_model(input, output) %>%
      keras::compile(optimizer = optimizer,
                     loss = loss,
                     metric = metric)
    model
  }


pow2_up_to <- function(n) {
  x <- floor(log2(n))
  powers_of_2(x)
}


powers_of_2 <- function(x) {
  x <- as.integer(x)
  l <- seq(1L, x)
  vapply(l, function(p) 2^p, 0)
}


pow2_range <- function(min, max) {
  min    <- nearest_pow2(min)
  powers <- pow2_up_to(max(min, max))
  idx    <- which(powers == min)

  powers[idx:length(powers)]
}


nearest_pow2 <- function(n) {
  fl <- floor(log2(n))
  cl <- ceiling(log2(n))

  pows <- powers_of_2(cl)
  lo <- pows[fl] %>% {abs(n - .)}
  hi <- pows[cl] %>% {abs(n - .)}

  if (lo > hi)
    return(pows[cl])
  else
    return(pows[fl])
}


