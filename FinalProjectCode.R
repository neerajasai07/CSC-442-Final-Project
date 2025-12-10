
library(tidyverse)
library(tibble)
library(ggplot2)

## Data Cleaning
spotify_tracks <- read.csv("spotify_tracks.csv")

spotify_tracks_cleaned <- spotify_tracks |>
  select(popularity, danceability, energy, loudness, speechiness, acousticness, 
         instrumentalness, liveness, valence, tempo)

# Check if columns have NA values
colSums(is.na(spotify_tracks_cleaned))

glimpse(spotify_tracks_cleaned)


## Histogram
ggplot(spotify_tracks_cleaned, aes(x = popularity)) +
  geom_histogram(bins = 30, color = "white") +
  labs(
    title = "Distrubution of Song Popularity", 
    x = "Popularity Score", 
    y = "Number of Tracks"
  )


## Scatterplots
ggplot(spotify_tracks_cleaned, aes(danceability, popularity)) +
  geom_point(alpha = 0.5) +
  geom_smooth() +
  labs (
    title = "Popularity Vs. Danceability", 
    x = "Danceability", 
    y = "Popularity"
  )

ggplot(spotify_tracks_cleaned, aes(energy, popularity)) +
  geom_point(alpha = 0.5) +
  geom_smooth() +
  labs (
    title = "Popularity Vs. Energy", 
    x = "Energy", 
    y = "Popularity"
  )

ggplot(spotify_tracks_cleaned, aes(acousticness, popularity)) +
  geom_point(alpha = 0.5) +
  geom_smooth() +
  labs (
    title = "Popularity Vs. Acousticness", 
    x = "Acousticness", 
    y = "Popularity"
  )

ggplot(spotify_tracks_cleaned, aes(valence, popularity)) +
  geom_point(alpha = 0.5) +
  geom_smooth() +
  labs (
    title = "Popularity Vs. Valence", 
    x = "Valence", 
    y = "Popularity"
  )


## PCA (Principal Component Analysis)
spotify_tracks_scaled <- scale(spotify_tracks_cleaned)
spotify_tracks_pca <- prcomp(spotify_tracks_scaled, center = TRUE, scale = TRUE)
spotify_tracks_pca
summary(spotify_tracks_pca)

eigenvalues <- spotify_tracks_pca$sdev^2

plot(eigenvalues/sum(eigenvalues), type = "b",
     main = "Scree Plot",
     xlab = "Principal Component",
     ylab = "Eigenvalue")


## K-Means Clustering
spotify_tracks_transformed <- data.frame(spotify_tracks_pca$x[, 1:2])
set.seed(123)
spotify_tracks_km <- kmeans(spotify_tracks_transformed, centers = 3, 
                            iter.max = 50, nstart = 20) 

spotify_tracks_transformed$Cluster <- factor(spotify_tracks_km$cluster)
ggplot(spotify_tracks_transformed, aes(x = PC1, PC2, color = Cluster)) +
  geom_point() +
  labs (
    title = "K-Means Clustering (k = 3) of PCA Scores", 
    x = "PC1", 
    y = "PC2"
  )


## Linear Regression Model
set.seed(123)
train_idx <- 1:floor(0.7*nrow(spotify_tracks_cleaned))
spotify_tracks_train <- spotify_tracks_cleaned[train_idx, ]
spotify_tracks_test <- spotify_tracks_cleaned[-train_idx,]

spotify_tracks_mod <- lm(popularity ~ danceability + energy + loudness + 
                           speechiness + acousticness + instrumentalness + 
                           liveness + valence + tempo, data = spotify_tracks_train)
summary(spotify_tracks_mod)

linear_pred <- predict(spotify_tracks_mod, newdata = spotify_tracks_test)

# Mean Absolute Error
linear_mae <- mean(abs(linear_pred - spotify_tracks_test$popularity))
sprintf("Mean absolute error: %f", linear_mae)


## Ridge Regression
library(glmnet)

set.seed(123)

train_idx <- 1:floor(0.7*nrow(spotify_tracks_cleaned))
spotify_tracks_train <- spotify_tracks_cleaned[train_idx, ]
spotify_tracks_test <- spotify_tracks_cleaned[-train_idx,]

x_train <- as.matrix(spotify_tracks_train[, -1])
y_train <- as.matrix(spotify_tracks_train$popularity)

x_test <- as.matrix(spotify_tracks_test[, -1])
y_test <- as.matrix(spotify_tracks_test$popularity)

ridge_reg <- glmnet(x_train, y_train, alpha = 0)
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0)
plot(cv_ridge)

optimal_lambda <- cv_ridge$lambda.min
sprintf("Optimal lambda: %f", optimal_lambda)

ridge_reg_2 <- glmnet(x_train, y_train, alpha = 0, lambda = optimal_lambda)

ridge_pred <- predict(ridge_reg_2, newx = x_test, s = optimal_lambda)

# Mean Absolute Error
ridge_mae <- mean(abs(ridge_pred - y_test))
sprintf("Mean absolute error: %f", ridge_mae)