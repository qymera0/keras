library(keras)
use_condaenv("r-reticulate", required = FALSE)
library(reticulate)
library(tfhub)
library(tfds)
library(tfdatasets)

# DOWNLOAD DATA -----------------------------------------------------------

imdb <-
        tfds_load(
                "imdb_reviews:1.0.0",
                split = list("train[:60%]", 
                             "train[-40%:]",
                             "test"),
                as_supervised = TRUE
        )


