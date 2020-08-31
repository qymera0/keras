library(keras)
library(tidyverse)


# LOAD THE DATA -----------------------------------------------------------

df <- read_csv("Basic ML with keras/movie_review.csv")

# EXPLORE THE DATA --------------------------------------------------------

df %>% count(tag)

# Dataset split

training_id <- sample.int(nrow(df), size = nrow(df)*0.8)

training <- df[training_id, ]

testing <- df[-training_id, ]

# Distribution of number of words

df$text %>% 
        strsplit(" ") %>% 
        sapply(length) %>% 
        summary()

# PREPARE THE DATA --------------------------------------------------------

# Text vectorization layer

num_words <- 10000

max_length <- 50

text_vectorization <-
        layer_text_vectorization(
                max_tokens = num_words,
                output_sequence_length = max_length
        )

# Adapt the layer to the data

text_vectorization %>% adapt(df$text)

get_vocabulary(text_vectorization)

text_vectorization(matrix(df$text[1], ncol = 1))

# BUILD THE MODEL ---------------------------------------------------------

input <- layer_input(shape = c(1), dtype = "string")

output <-
        input %>% 
        text_vectorization() %>% 
        layer_embedding(input_dim = num_words + 1,
                        output_dim = 16) %>% 
        layer_global_average_pooling_1d() %>% 
        layer_dense(units = 16, activation = "relu") %>% 
        layer_dropout(0.5) %>% 
        layer_dense(units = 1, activation = "sigmoid")

model <- keras_model(input, output)

# Model compilation

model %>% 
        compile(
                optimizer = "adam",
                loss = 'binary_crossentropy',
                metrics = list("accuracy")
        )

# TRAIN MODEL -------------------------------------------------------------

history <-
        model %>% 
        fit(
                training$text,
                as.numeric(training$tag == "pos"),
                epochs = 10,
                barch_size = 512,
                validation_split = 0.2,
                verbose = 2
        )


# EVALUATE THE MODEL ------------------------------------------------------

results <- model %>% evaluate(testing$text,
                             as.numeric(testing$tag == "pos"),
                             verbose = 0)

plot(history)
