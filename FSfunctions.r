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

CIFE <- function(X, Y, beta = 0.5, k = ncol(X)) {
  # Convert input keeping native column order
  X <- as.data.frame(X)
  Y <- as.factor(Y)
  
  # Get features in original order
  features <- colnames(X)
  
  # Initialize results
  scores <- setNames(rep(NA, ncol(X)), features)
  selected <- character(0)
  
  # Precompute entropy
  H_Y <- entropy(Y)
  
  for (i in 1:k) {
    remaining <- setdiff(features, selected)
    
    # Calculate CIFE criterion: I(f;Y) - β[I(f;S) - I(f;S|Y)]
    if (length(selected) > 0) {
      redundancy <- sapply(remaining, function(f) {
        sum(sapply(selected, function(s) {
          # I(f;S) - I(f;S|Y)
          mi_fs <- mutinformation(discretize(X[[f]]), discretize(X[[s]])) / H_Y
          cmi_fs_y <- (mutinformation(discretize(X[[f]]), discretize(X[[s]])) - 
                         condinformation(discretize(X[[f]]), discretize(X[[s]]), Y)) / H_Y
          mi_fs - cmi_fs_y
        }))
      })
    } else {
      redundancy <- setNames(rep(0, length(remaining)), remaining)
    }
    
    # Calculate relevance I(f;Y)
    relevance <- sapply(remaining, function(f) {
      mutinformation(discretize(X[[f]]), Y) / H_Y
    })
    
    # CIFE score
    current_scores <- relevance - beta * redundancy
    
    # Select best feature without reordering
    best_idx <- which.max(current_scores)
    best_feature <- names(current_scores)[best_idx]
    scores[best_feature] <- current_scores[best_idx]
    selected <- c(selected, best_feature)
  }
  
  # Fill NA with 0 without reordering
  scores[is.na(scores)] <- 0
  
  # Return in original feature order
  list(
    selection = setNames(seq_along(features), features),
    score = scores
  )
}

library(infotheo)
library(praznik)

DMIM <- function(X, Y, k = ncol(X)) {

  # Convert input to proper formats
  X <- as.data.frame(X)
  Y <- as.factor(Y)
  feature_names <- colnames(X)
  
  # Initialize results storage
  selected_features <- character(0)
  remaining_features <- feature_names
  feature_scores <- setNames(rep(0, length(feature_names)), feature_names)
  
  # Precompute target entropy for normalization
  H_Y <- entropy(Y)
  
  # DMIM feature selection process
  for (i in 1:k) {
    # Calculate mutual information with target for remaining features
    mi_target <- sapply(remaining_features, function(f) {
      x_disc <- discretize(X[[f]])
      mutinformation(x_disc, Y) / H_Y  # Normalized MI
    })
    
    # Calculate dynamic redundancy term
    redundancy <- if (length(selected_features) > 0) {
      sapply(remaining_features, function(f) {
        max(sapply(selected_features, function(s) {
          # Dynamic term: I(f;S) - I(f;Y|S)
          x_f <- discretize(X[[f]])
          x_s <- discretize(X[[s]])
          
          mi_fs <- mutinformation(x_f, x_s) / H_Y
          cmi_fs_y <- condinformation(x_f, Y, x_s) / H_Y
          
          mi_fs - cmi_fs_y  # Core DMIM term
        }))
      })
    } else {
      rep(0, length(remaining_features))
    }
    
    # Calculate DMIM criterion scores
    dmim_scores <- mi_target - redundancy
    
    # Select feature with maximum DMIM score
    best_idx <- which.max(dmim_scores)
    best_feature <- remaining_features[best_idx]
    
    # Update results
    feature_scores[best_feature] <- dmim_scores[best_idx]
    selected_features <- c(selected_features, best_feature)
    remaining_features <- setdiff(remaining_features, best_feature)
  }
  
  # Return results in consistent format (unsorted)
  list(
    selection = setNames(match(feature_names, feature_names), feature_names),
    score = feature_scores
  )
}