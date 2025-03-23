# Seed
set.seed(123)

# Read the datasets
file1 <- "../image+segmentation/segmentation.data"
if (!file.exists(file1) )
      stop("segmentation.data file does not exist")
file2 <- "../image+segmentation/segmentation.test"
if (!file.exists(file2) )
      stop("segmentation.test file does not exist")

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
