if (!require("FSelectorRcpp")) install.packages("FSelectorRcpp", dependencies=TRUE)
if (!require("praznik")) install.packages("praznik", dependencies=TRUE)
if (!require("dplyr")) install.packages("dplyr", dependencies=TRUE)
if (!require("infotheo")) install.packages("infotheo", dependencies=TRUE)
if (!require("mRMRe")) install.packages("mRMRe", dependencies=TRUE)
if (!require("class")) install.packages("class", dependencies = TRUE)
#install.packages("FSelectorRcpp")

library(FSelectorRcpp)
library(praznik)
library(dplyr)
library(infotheo)
library(mRMRe)
library(class)

# Set working directory
# getwd()
# setwd("...");

# Seed
set.seed(123)

# Import implemented FS functions
source('FSfunctions.r')

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

# Convert the target variable to factor
train_data$Class <- as.factor(train_data$Class)

# Get names of variables
feature_names <- colnames(train_data)[-1]

# Calculate mutual information between each feature and target variable
mi_values <- FSelectorRcpp::information_gain(Class ~ ., train_data)
print(mi_values)

# Convert to lists
X <- train_data[, feature_names]
Y <- train_data$Class

# Selected features list
S <- c()

#----------------------------------
# Forward feature selection methods
#----------------------------------

# MIM (Mutual Information Maximization)
# mim_ranking <- mi_values %>% arrange(desc(importance))
mim_ranking <- praznik::MIM(X, Y, k = length(feature_names))
print("MIM Ranking:")
print(mim_ranking)

# MIFS (Mutual Information Feature Selection)
# already sorted by scores
#-------------------------------------------------------
mifs_ranking <- MIFS(X, Y, 0.5, k = length(feature_names))
print("MIFS Ranking:")
print(mifs_ranking)

# mRMR (Minimum Redundancy Maximum Relevance)
mrmr_ranking <- praznik::MRMR(X, Y, k = length(feature_names))
print("mRMR Ranking:")
print(mrmr_ranking)

# maxMIFS (Máxima Penalización de Información Mutua)
# already sorted by scores
#-------------------------------------------------------
max_mifs_ranking <- maxMIFS(X, Y, 0.5, k = length(feature_names))
print("maxMIFS Ranking:")
print(max_mifs_ranking)

# CIFE (Conditional Infomax Feature Extraction)
#-------------------------------------------------------
#cife_ranking <- praznik::CIFE(X, Y, k = length(feature_names))

# JMI (Joint Mutual Information)
jmi_ranking <- praznik::JMI(X, Y, k = length(feature_names))
print("JMI Ranking:")
print(jmi_ranking)

# CMIM (Conditional Mutual Information Maximization)
cmim_ranking <- praznik::CMIM(X, Y, k = length(feature_names))
print("CMIM Ranking:")
print(cmim_ranking)

# JMIM (Joint Mutual Information Maximization)
jmim_ranking <- praznik::JMIM(X, Y, k = length(feature_names))
print("JMIM Ranking:")
print(jmim_ranking)

# DMIM (Dynamic Mutual Information Maximization)
#-------------------------------------------------------
# dmim_ranking <- mi_values %>%
#   mutate(score = MI - sapply(feature, function(x) max(FSelectorRcpp::mutual_information(X[[x]], X))) +
#            sapply(feature, function(x) max(FSelectorRcpp::conditional_mutual_information(X[[x]], X, Y)))) %>%
#   arrange(desc(score))

# Sort features by importance

process_and_predict <- function(mim_sorted, train_data, test_data, k = 3) {
  # Extract top 10 and top 20 features
  top_10_features <- names(sort(mim_sorted, decreasing = TRUE))[1:10]
  top_20_features <- names(sort(mim_sorted, decreasing = TRUE))[1:20]
  
  # Eliminate NA values
  top_10_features_clean <- top_10_features[!is.na(top_10_features)]
  top_20_features_clean <- top_20_features[!is.na(top_20_features)]
  
  # Select data based on the top features
  train_data_10 <- train_data[, c("Class", top_10_features_clean)]
  test_data_10 <- test_data[, c("Class", top_10_features_clean)]
  
  train_data_20 <- train_data[, c("Class", top_20_features_clean)]
  test_data_20 <- test_data[, c("Class", top_20_features_clean)]
  
  # Apply k-NN for top 10 features
  pred_10 <- knn(train = train_data_10[,-1], test = test_data_10[,-1], 
                 cl = train_data_10$Class, k = k)
  
  # Apply k-NN for top 20 features
  pred_20 <- knn(train = train_data_20[,-1], test = test_data_20[,-1], 
                 cl = train_data_20$Class, k = k)
  
  # Store predictions in the test data
  test_data_10$Predicted <- pred_10
  test_data_20$Predicted <- pred_20
  
  # Display predictions
  cat("Predictions with the top 10 most important features:\n")
  print(table(test_data_10$Class, test_data_10$Predicted))
  
  cat("\nPredictions with the top 20 most important features:\n")
  print(table(test_data_20$Class, test_data_20$Predicted))
  
  # Return the processed test data as output
  return(list(test_data_10 = test_data_10, test_data_20 = test_data_20))
}


# Process and predict
mim_sorted <- mim_ranking$score
print(mim_sorted)
results <- process_and_predict(mim_sorted, train_data, test_data, k = 3)

mifs_sorted <- mifs_ranking$score
print(mifs_sorted)
results <- process_and_predict(mifs_sorted, train_data, test_data, k = 3)

mrmr_sorted <- mrmr_ranking$score
print(mrmr_sorted)
results <- process_and_predict(mrmr_sorted, train_data, test_data, k = 3)

max_mifs_sorted <- max_mifs_ranking$score
print(max_mifs_sorted)
results <- process_and_predict(max_mifs_sorted, train_data, test_data, k = 3)

jmi_sorted <- jmi_ranking$score
print(jmi_sorted)
results <- process_and_predict(jmi_sorted, train_data, test_data, k = 3)

cmim_sorted <- cmim_ranking$score
print(cmim_sorted)
results <- process_and_predict(cmim_sorted, train_data, test_data, k = 3)

jmim_sorted <- jmim_ranking$score
print(jmim_sorted)
results <- process_and_predict(jmim_sorted, train_data, test_data, k = 3)
