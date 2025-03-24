# Seed
set.seed(123)

# Read the datasets
file1 <- "./image_segmentation/segmentation.data"
if (!file.exists(file1)) { 
  stop("segmentation.data file does not exist") 
} else { 
  print("segmentation.data found") 
}
file2 <- "./image_segmentation/segmentation.test"
if (!file.exists(file2)) { 
  stop("segmentation.test file does not exist") 
} else { 
  print("segmentation.test found") 
}

data1 <- read.csv(file1, skip = 5, header = FALSE)
data2 <- read.csv(file2, skip = 5, header = FALSE)

# Define column names 
column_names <- c("Class", "REGION-CENTROID-COL", "REGION-CENTROID-ROW", "REGION-PIXEL-COUNT",
                  "SHORT-LINE-DENSITY-5", "SHORT-LINE-DENSITY-2", "VEDGE-MEAN", "VEDGE-SD",
                  "HEDGE-MEAN", "HEDGE-SD", "INTENSITY-MEAN", "RAWRED-MEAN", "RAWBLUE-MEAN",
                  "RAWGREEN-MEAN", "EXRED-MEAN", "EXBLUE-MEAN", "EXGREEN-MEAN", "VALUE-MEAN",
                  "SATURATION-MEAN", "HUE-MEAN")

colnames(data1) <- column_names
colnames(data2) <- column_names

# Combine the datasets
data <- rbind(data1, data2)

# Shuffle the data
shuffled_data <- data[sample(nrow(data)), ]

# Split the data: 70% training, 30% testing
train_index <- 1:floor(0.7 * nrow(shuffled_data))
train_data <- shuffled_data[train_index, ]
test_data <- shuffled_data[-train_index, ]

cat("Training instances:", nrow(train_data), "\n")
cat("Testing instances:", nrow(test_data), "\n")

# Exercise 3: Feature Selection
library(FSelector)

# Convert the target variable to factor
train_data$Class <- as.factor(train_data$Class)

# Apply DMIM for feature selection
feature_scores <- information.gain(Class ~ ., train_data)

# Sort features by importance
feature_scores <- data.frame(attributes = rownames(feature_scores), importance = feature_scores$attr_importance)
top_features <- feature_scores[order(-feature_scores$importance), ]

# Display the top 10 and top 20 features
print("Top 10 features:")
print(top_features[1:10, ])

print("Top 20 features:")
print(top_features[1:20, ])

# Exercise 5a: Train k-NN with k = 3
library(class)

# Select top 10 and top 20 features for training and testing
top_10_features <- as.character(top_features$attributes[1:10])
top_20_features <- as.character(top_features$attributes[1:20])

# Omit NA in tables
top_10_features <- na.omit(top_10_features)
top_20_features <- na.omit(top_20_features)

train_data_10 <- train_data[, c("Class", top_10_features)]
test_data_10 <- test_data[, c("Class", top_10_features)]

train_data_20 <- train_data[, c("Class", top_20_features)]
test_data_20 <- test_data[, c("Class", top_20_features)]

# Train and predict using k-NN (k=3)
k <- 3
pred_10 <- knn(train = train_data_10[,-1], test = test_data_10[,-1], 
               cl = train_data_10$Class, k = k)
pred_20 <- knn(train = train_data_20[,-1], test = test_data_20[,-1], 
               cl = train_data_20$Class, k = k)

# Store predictions
test_data_10$Predicted <- pred_10
test_data_20$Predicted <- pred_20

print("Predictions with top 10 features:")
print(table(test_data_10$Class, test_data_10$Predicted))

print("Predictions with top 20 features:")
print(table(test_data_20$Class, test_data_20$Predicted))
