library(infotheo)
library(praznik)

MIFS <- function(X, Y, beta = 0.5, k = ncol(X)) {
  # Convert input to data frame and factor
  X <- as.data.frame(X)
  Y <- as.factor(Y)
  
  # Get feature names in original order
  feature_names <- colnames(X)
  
  # Initialize results
  selected <- character(0)
  scores <- setNames(numeric(length(feature_names)), feature_names)
  
  # Precompute entropy
  H_Y <- entropy(Y)
  
  # Precompute mutual information with target
  mi_target <- sapply(X, function(x) {
    x_disc <- discretize(x)
    mutinformation(x_disc, Y) / H_Y
  })
  
  for (i in 1:k) {
    # Calculate redundancy for remaining features
    remaining <- setdiff(feature_names, selected)
    redundancy <- if (length(selected) > 0) {
      sapply(remaining, function(f) {
        mean(sapply(selected, function(s) {
          mutinformation(discretize(X[[f]]), discretize(X[[s]])) / H_Y
        }))
      })
    } else {
      setNames(rep(0, length(remaining)), remaining)
    }
    
    # Calculate scores
    current_scores <- mi_target[remaining] - beta * redundancy
    
    # Select best feature
    best_feature <- names(which.max(current_scores))
    scores[best_feature] <- current_scores[best_feature]
    selected <- c(selected, best_feature)
  }
  
  list(
    selection = structure(match(feature_names, feature_names), 
                          names = feature_names),
    score = scores
  )
}


maxMIFS <- function(X, Y, beta = 0.5, k = ncol(X)) {
  # Convert input keeping native column order
  X <- as.data.frame(X)
  Y <- as.factor(Y)
  
  # Get features in whatever order they come
  features <- colnames(X)
  
  # Initialize in native order
  scores <- setNames(rep(NA, ncol(X)), features)
  selected <- character(0)
  
  # Precompute entropy
  H_Y <- entropy(Y)
  
  for (i in 1:k) {
    remaining <- setdiff(features, selected)
    
    # Calculate maximum redundancy (maxMIFS difference)
    redundancy <- if (length(selected) > 0) {
      sapply(remaining, function(f) {
        max(sapply(selected, function(s) {
          mutinformation(discretize(X[[f]]), discretize(X[[s]])) / H_Y
        }))
      })
    } else {
      setNames(rep(0, length(remaining)), remaining)
    }
    
    # Get scores in native order
    mi_scores <- sapply(remaining, function(f) {
      mutinformation(discretize(X[[f]]), Y) / H_Y
    })
    
    current_scores <- mi_scores - beta * redundancy
    
    # Select without reordering
    best_idx <- which.max(current_scores)
    best_feature <- names(current_scores)[best_idx]
    scores[best_feature] <- current_scores[best_idx]
    selected <- c(selected, best_feature)
  }
  
  # Fill NA with 0 without reordering
  scores[is.na(scores)] <- 0
  
  list(
    selection = setNames(seq_along(features), features),
    score = scores
  )
}
