library(keras)
library(tidyverse)
library(tfdatasets)

# BOSTON HOUSING PRICES DATASET ------------------------------------------

boston_housing <- dataset_boston_housing()

c(train_data, train_labels) %<-% boston_housing$train

c(test_data, test_labels) %<-% boston_housing$test

paste0("Training entries: ", length(train_data), ", labels: ", length(train_labels))

train_data[1, ]

# Change column names

column_names <- c('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                  'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT')

train_df <-
        train_data %>% 
        as_tibble(.name_repair = "minimal") %>% 
        setNames(column_names) %>% 
        mutate(label = train_labels)

test_df <-
        test_data %>% 
        as_tibble(.name_repair = "minimal") %>% 
        setNames(column_names) %>% 
        mutate(label = test_labels)

# NORMALIZE FEATURES ------------------------------------------------------

spec <-
        feature_spec(train_df, label ~ .) %>% 
        step_numeric_column(all_numeric(),
                            normalizer_fn = scaler_standard()) %>% 
        fit()

spec

layer <-
        layer_dense_features(
                
                feature_columns = dense_features(spec),
                dtype = tf$float32
                
        )

layer(train_df)

# CREATE THE MODEL --------------------------------------------------------

input <- layer_input_from_dataset(train_df %>% 
                                          select(-label))

output <-
        input %>% 
        layer_dense_features(dense_features(spec)) %>% 
        layer_dense(units = 64, activation = "relu") %>%
        layer_dense(units = 64, activation = "relu") %>%
        layer_dense(units = 1) 

model <- keras_model(input, output)

summary(model)

# COMPILE MODEL ----------------------------------------------------------

model %>% 
        compile(
                loss = "mse",
                optimizer = optimizer_rmsprop(),
                metrics = list("mean_absolute_error")
        )


# Wrap the model inside the function to allow experimentation

build_model <- function(){
        
        input <- layer_input_from_dataset(train_df %>% 
                                                  select(-label))
        
        output <- input %>% 
                layer_dense_features(dense_features(spec)) %>% 
                layer_dense(units = 64, activation = "relu") %>%
                layer_dense(units = 64, activation = "relu") %>%
                layer_dense(units = 1) 
        
        model <- keras_model(input, output)
        
        model %>% 
                compile(
                        loss = "mse",
                        optimizer = optimizer_rmsprop(),
                        metrics = list("mean_absolute_error")
                )
        
        model
        
}


# TRAIN THE MODEL ---------------------------------------------------------

# Display training progress by printing a single dot for each completed epoch.

print_dot_callback <-
        callback_lambda(
                on_epoch_end = function(epoch, logs){
                        if(epoch %% 80 == 0) cat("\n")
                        cat(".")
                }
        )

model <- build_model()

history <-
        model %>% 
        fit(
                x = train_df %>% select(-label),
                y = train_df$label,
                epochs = 500,
                validation_split = 0.2,
                verbose = 0,
                callbacks = list(print_dot_callback)
        )

plot(history)

c(loss, mae) %<-% (model %>% evaluate(test_df %>% select(-label), test_df$label, verbose = 0))

paste0("Mean absolute error on test set: $", sprintf("%.2f", mae * 1000))


# PREDICTION --------------------------------------------------------------

test_predictions <- model %>% predict(test_df %>% select(-label))

test_predictions[ , 1]
