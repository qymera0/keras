library(keras)
library(tidyverse)

# IMPORT DATASET ---------------------------------------------------------

fashionMinst <- dataset_fashion_mnist()

c(train_images, train_labels) %<-% fashionMinst$train

c(test_images, test_labels) %<-% fashionMinst$train

# Store class labels

class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')

# EXPLORE THE DATA --------------------------------------------------------

dim(train_images)

dim(train_labels)

train_labels[1:20]

# PREPROCESS THE DATA -----------------------------------------------------

image1 <- as.data.frame(train_images[1, , ])

colnames(image1) <- seq_len(ncol(image1))

image1$y <- seq_len(nrow(image1))

image1 <- gather(image1, "x", "value", -y)

image1$x <- as.integer(image1$x)

ggplot(image1, aes(x = x, y = y, fill = value)) +
       geom_tile() +
       scale_fill_gradient(low = "white", high = "black", na.value = NA) +
       scale_y_reverse() +
       theme_minimal() +
       theme(panel.grid = element_blank())   +
       theme(aspect.ratio = 1) +
       xlab("") +
       ylab("")


# separate training and test data sets

train_Images <- train_images / 255

test_Images <- test_images / 255

par(mfcol=c(5,5))

par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')

for (i in 1:25) { 
        img <- train_images[i, , ]
        img <- t(apply(img, 2, rev)) 
        image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
              main = paste(class_names[train_labels[i] + 1]))
}


# BUILD THE MODEL ---------------------------------------------------------


# Setup the layers

model <-
        keras_model_sequential() %>% 
        layer_flatten(input_shape = c(28,28)) %>% 
        layer_dense(units = 128, activation = "relu") %>% 
        layer_dense(units = 10, activation = "softmax")

# Copile the model

model %>% 
        compile(
                optimizer = "adam",
                loss = "sparse_categorical_crossentropy",
                metrics = c('accuracy')
        )

# Train the modes

model %>% fit(train_Images, train_labels, epochs = 5, verbose = 2)

# MODEL ACCURACY ----------------------------------------------------------

score <- model %>% evaluate(test_Images, test_labels, verbose = 0)

cat("Test loss:", score$loss, "\n")

cat('Test accuracy:', score$acc, "\n")

# MAKE PREDICTIONS --------------------------------------------------------

predictions <- model %>% predict(test_images)

predictions[1, ]

which.max(predictions[1, ])

class_pred <- model %>% predict_classes(test_images)

class_pred[1:20]

test_labels[1]


par(mfcol=c(5,5))

par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')

for (i in 1:25) { 
        img <- test_images[i, , ]
        img <- t(apply(img, 2, rev)) 
        # subtract 1 as labels go from 0 to 9
        predicted_label <- which.max(predictions[i, ]) - 1
        true_label <- test_labels[i]
        if (predicted_label == true_label) {
                color <- '#008800' 
        } else {
                color <- '#bb0000'
        }
        image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
              main = paste0(class_names[predicted_label + 1], " (",
                            class_names[true_label + 1], ")"),
              col.main = color)
}

img <- test_images[1, , , drop = FALSE]

dim(img)

predictions <- model %>% predict(img)

predictions
