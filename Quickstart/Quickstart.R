library(keras)
library(tensorflow)

mnist <- dataset_mnist()

# Converts pixels (0 to 255) to float 0 to 1

mnist$train$x <- mnist$train$x/255

mnist$test$x <- mnist$test$x/255

# Keras model definition

model <-
        keras_model_sequential() %>% 
        layer_flatten(input_shape = c(28,28)) %>% 
        layer_dense(units = 128, activation = "relu") %>% 
        layer_dropout(0.2) %>% 
        layer_dense(10, activation = "softmax")

summary(model)

# Compilation (loss and optimizer)

model %>% 
        compile(
                loss = "sparse_categorical_crossentropy",
                optimizer = "adam",
                metrics = "accuracy"
        )

# Fit

model %>% 
        fit(
                x = mnist$train$x,
                y = mnist$train$y,
                epochs = 5,
                validation_split = 0.3,
                verbose = 2
        )

# Predictions

predictions <- predict(model, mnist$test$x)

head(predictions)

# Evaluate model

model %>% 
        evaluate(mnist$test$x, mnist$test$y, verbose = 0)

# Save model

setwd("~/R/Learning/keras/Quickstart")

save_model_tf(object = model, filepath = "model")

# reload model

reloaded_model <- load_model_tf("model")

all.equal(predict(model, mnist$test$x), predict(reloaded_model, mnist$test$x))
