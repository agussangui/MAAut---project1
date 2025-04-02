if (!require("FSelectorRcpp")) install.packages("FSelectorRcpp", dependencies=TRUE)
if (!require("praznik")) install.packages("praznik", dependencies=TRUE)
if (!require("dplyr")) install.packages("dplyr", dependencies=TRUE)
if (!require("infotheo")) install.packages("infotheo", dependencies=TRUE)
if (!require("mRMRe")) install.packages("mRMRe", dependencies=TRUE)
if (!require("class")) install.packages("class", dependencies=TRUE)
if (!require("caret")) install.packages("caret", dependencies=TRUE)
if (!require("GGally")) install.packages("GGally", dependencies=TRUE)

library(FSelector)
library(praznik)
library(dplyr)
library(infotheo)
library(mRMRe)
library(class)
library(caret)
library(GGally)

# Set working directory
# getwd()
# setwd("...");

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
#-------------------------------------------------------


# mRMR (Minimum Redundancy Maximum Relevance)
mrmr_ranking <- praznik::MRMR(X, Y, k = length(feature_names))
print("mRMR Ranking:")
print(mrmr_ranking)

# maxMIFS (Máxima Penalización de Información Mutua)
#-------------------------------------------------------
# max_mifs_ranking <- mi_values %>%
#   mutate(score = MI - sapply(feature, function(x) max(FSelectorRcpp::mutual_information(X[[x]], X)))) %>%
#   arrange(desc(score))

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


# Process and predict MIM
mim_sorted <- mim_ranking$score
print(mim_sorted)
mim_results <- process_and_predict(mim_sorted, train_data, test_data, k = 3)

# Process and predict mRMR
mrmr_sorted <- mrmr_ranking$score
print(mrmr_sorted)
mrmr_results <- process_and_predict(mrmr_sorted, train_data, test_data, k = 3)

# Process and predict JMI
jmi_sorted <- jmi_ranking$score
print(jmi_sorted)
jmi_results <- process_and_predict(jmi_sorted, train_data, test_data, k = 3)

# Process and predict CMIM
cmim_sorted <- cmim_ranking$score
print(cmim_sorted)
cmim_results <- process_and_predict(cmim_sorted, train_data, test_data, k = 3)

# Process and predict JMIM
jmim_sorted <- jmim_ranking$score
print(jmim_sorted)
jmim_results <- process_and_predict(jmim_sorted, train_data, test_data, k = 3)

# Function for metrics
calculate_metrics <- function(test_data) {
  confusion_mat <- confusionMatrix(as.factor(test_data$Predicted), as.factor(test_data$Class))
  
  # Accuracy
  accuracy <- confusion_mat$overall["Accuracy"]
  
  # Macro Precision, Recall y F1
  precision <- mean(confusion_mat$byClass[, "Precision"], na.rm = TRUE)
  recall <- mean(confusion_mat$byClass[, "Recall"], na.rm = TRUE)
  f1 <- mean(confusion_mat$byClass[, "F1"], na.rm = TRUE)
  
  cat("\nPerformance Metrics:\n")
  cat("Accuracy:", accuracy, "\n")
  cat("Macro Precision:", precision, "\n")
  cat("Macro Recall:", recall, "\n")
  cat("Macro F1:", f1, "\n")
  
  return(list(Accuracy = accuracy, Macro_Precision = precision, Macro_Recall = recall, Macro_F1 = f1))
}

# MIM metrics
metrics_mim_10 <- calculate_metrics(mim_results$test_data_10)
metrics_mim_20 <- calculate_metrics(mim_results$test_data_20)

# mRMR metrics
metrics_mrmr_10 <- calculate_metrics(mrmr_results$test_data_10)
metrics_mrmr_20 <- calculate_metrics(mrmr_results$test_data_20)

# JMI metrics
metrics_jmi_10 <- calculate_metrics(jmi_results$test_data_10)
metrics_jmi_20 <- calculate_metrics(jmi_results$test_data_20)

# CMIM metrics
metrics_cmim_10 <- calculate_metrics(cmim_results$test_data_10)
metrics_cmim_20 <- calculate_metrics(cmim_results$test_data_20)

# JMIM metrics
metrics_jmim_10 <- calculate_metrics(jmim_results$test_data_10)
metrics_jmim_20 <- calculate_metrics(jmim_results$test_data_20)

# Remove constant columns (zero variance)
non_constant_columns <- apply(train_data[, -1], 2, function(col) sd(col) != 0)
train_data_filtered <- train_data[, c(TRUE, non_constant_columns)]  # Keep Class column

# Apply PCA
pca_model <- prcomp(train_data_filtered[, -1], center = TRUE, scale. = TRUE)

# Analyze the explained variance
explained_variance <- summary(pca_model)
print(explained_variance$importance)

dev.new()
# Plot the cumulative variance
plot(cumsum(explained_variance$importance[2,]), type="b", 
     xlab="Number of Components", ylab="Cumulative Variance",
     main="Variance Explained by Principal Components")

# Select the number of components that capture at least 95% of the variance
num_components <- which(cumsum(explained_variance$importance[2,]) >= 0.95)[1]
cat("Number of selected components:", num_components, "\n")

# Transform training and test data using only the selected components
train_pca <- data.frame(Class = train_data$Class, pca_model$x[, 1:num_components])
test_pca <- data.frame(Class = test_data$Class, predict(pca_model, test_data[, -1])[, 1:num_components])

# Apply k-NN with the transformed data
pred_pca <- knn(train = train_pca[, -1], test = test_pca[, -1], 
                cl = train_pca$Class, k = 3)

# Evaluate the model with PCA
test_pca$Predicted <- pred_pca
metrics_pca <- calculate_metrics(test_pca)

# Generate PAIR PLOTS and save it
plot_data <- cbind(
  train_data[, sapply(train_data, is.numeric)],
  Class = train_data$Class
)

# Get color palette
class_levels <- levels(plot_data$Class)
color_palette <- RColorBrewer::brewer.pal(length(class_levels), "Set1")
names(color_palette) <- class_levels

create_plot_matrix <- function(data, cols) {
  plot_list <- list()
  
  for(i in seq_along(cols)) {
    for(j in seq_along(cols)) {
      if(i == j) {
        # Diagonal: Density plots
        plot_list[[paste(i,j)]] <- ggplot2::ggplot(data, ggplot2::aes(x = .data[[cols[i]]], color = Class)) +
          ggplot2::geom_density(alpha = 0.4) +
          ggplot2::scale_color_manual(values = color_palette) +
          ggplot2::theme_minimal()
      } else if(i < j) {
        # Upper triangle: Combined correlations
        cor_text <- c(
          paste("Overall:", round(cor(data[[cols[i]]], data[[cols[j]]]), 2)),
          sapply(levels(data$Class), function(cls) {
            paste0(cls, ": ", round(cor(data[data$Class == cls, cols[i]], 
                                        data[data$Class == cls, cols[j]]), 2))
          })
        )
        
        plot_list[[paste(i,j)]] <- ggplot2::ggplot(data) +
          ggplot2::annotate(
            "text",
            x = 0.5,
            y = seq(0.9, 0.1, length.out = length(cor_text)),
            label = cor_text,
            size = c(3, rep(2.5, length(levels(data$Class)))),
            color = c("black", color_palette[levels(data$Class)]),
            fontface = c("bold", rep("plain", length(levels(data$Class))))
          ) +
          ggplot2::theme_void() +
          ggplot2::coord_cartesian(clip = "off")  # Allow text to extend beyond panel
      } else {
        # Lower triangle: Scatter plots
        plot_list[[paste(i,j)]] <- ggplot2::ggplot(data, ggplot2::aes(x = .data[[cols[i]]], y = .data[[cols[j]]], color = Class)) +
          ggplot2::geom_point(alpha = 0.3, size = 0.8) +
          ggplot2::scale_color_manual(values = color_palette) +
          ggplot2::theme_minimal()
      }
    }
  }
  
  # Combine plots with adjusted panel sizes
  combined <- patchwork::wrap_plots(plot_list, ncol = length(cols), guides = "collect",
                                    widths = rep(1, length(cols)), 
                                    heights = rep(1, length(cols))) +
    patchwork::plot_annotation(title = "Complete Pairs Plot (Colored by Class)") &
    ggplot2::theme(
      legend.position = "bottom",
      axis.text = ggplot2::element_text(size = 4),
      strip.text = ggplot2::element_text(size = 5),
      plot.margin = ggplot2::unit(c(0, 0, 0.5, 0), "cm")  # Add bottom margin
    )
  
  return(combined)
}

# Save the plot
ggplot2::ggsave(
  "PAIRS_PLOT.png",
  plot = create_plot_matrix(plot_data, colnames(plot_data)[1:19]),
  width = 40,
  height = 38,
  dpi = 200,
  limitsize = FALSE
)

# Clean up
rm(plot_data, color_palette, class_levels)
gc()