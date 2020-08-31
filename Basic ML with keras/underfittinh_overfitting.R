library(keras)
library(tidyverse)

num_words <- 1000

imdb <- dataset_imdb(num_words = num_words)

c(train_data, train_labels) %<-% imdb$train

c(test_data, test_labels) %<-% imdb$test

multi_hot_sequences <-
        function(sequences, dimension){
                
                multi_hot <- matrix(0, 
                                    nrow = length(sequences),
                                    ncol = dimension)
                for (i in 1:length(sequences)){
                        multi_hot[i, sequences[[i]]] <- 1
                }
                
                multi_hot
        }

train_data <- multi_hot_sequences(train_data, num_words)

test_data <- multi_hot_sequences(test_data, num_words)

first_text <- data.frame(word = 1:num_words, value = train_data[1, ])

ggplot(first_text, aes(x = word, y = value)) +
        geom_line() +
        theme(axis.title.y = element_blank(),
              axis.text.y = element_blank(),
              axis.ticks.y = element_blank())


# DEMONSTRATE OVERFITING --------------------------------------------------

# Baseline model

baseline_model <-
        keras_model_sequential() %>%
        layer_dense(units = 16,
                    activation = "relu",
                    input_shape = num_words) %>% 
        layer_dense(units = 16, 
                    activation = "relu") %>% 
        layer_dense(units = 1,
                    activation = "sigmoid")

baseline_model %>% 
        compile(
                optimizer = "adam",
                loss = "binary_crossentropy",
                metrics = list("accuracy")
        )

summary(baseline_model)

# fit the model

baseline_history <-
        baseline_model %>% 
        fit(
                train_data,
                train_labels,
                epochs = 20,
                batch_size = 512,
                validation_data = list(test_data, test_labels),
                verbose = 2
        )

# Smaller model

smaller_model <- 
        keras_model_sequential() %>%
        layer_dense(units = 4, 
                    activation = "relu", 
                    input_shape = num_words) %>%
        layer_dense(units = 4, 
                    activation = "relu") %>%
        layer_dense(units = 1, 
                    activation = "sigmoid")

smaller_model %>% compile(
        optimizer = "adam",
        loss = "binary_crossentropy",
        metrics = list("accuracy")
)

smaller_history <- 
        smaller_model %>% 
        fit(
                train_data,
                train_labels,
                epochs = 20,
                batch_size = 512,
                validation_data = list(test_data, test_labels),
                verbose = 2
        )

# Bigger Model

bigger_model <- 
        keras_model_sequential() %>%
        layer_dense(units = 512, 
                    activation = "relu", 
                    input_shape = num_words) %>%
        layer_dense(units = 512, 
                    activation = "relu") %>%
        layer_dense(units = 1,
                    activation = "sigmoid")

bigger_model %>% 
        compile(
                optimizer = "adam",
                loss = "binary_crossentropy",
                metrics = list("accuracy")
)

bigger_history <- 
        bigger_model %>% 
        fit(
                train_data,
                train_labels,
                epochs = 20,
                batch_size = 512,
                validation_data = list(test_data, test_labels),
                verbose = 2
)

# Plot the comparison

compare_cx <- data.frame(
        baseline_train = baseline_history$metrics$loss,
        baseline_val = baseline_history$metrics$val_loss,
        smaller_train = smaller_history$metrics$loss,
        smaller_val = smaller_history$metrics$val_loss,
        bigger_train = bigger_history$metrics$loss,
        bigger_val = bigger_history$metrics$val_loss
) %>%
        rownames_to_column() %>%
        mutate(rowname = as.integer(rowname)) %>%
        gather(key = "type", value = "value", -rowname)

ggplot(compare_cx, aes(x = rowname, y = value, color = type)) +
        geom_line() +
        xlab("epoch") +
        ylab("loss")

# Strategies

