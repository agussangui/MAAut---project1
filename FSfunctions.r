library(infotheo)
library(praznik)

calculate_k <- function(n, type = "u", c = NULL) {
  if (type == "u") {
    ceiling(sqrt(n))
  } else if (type == "j") {
    ceiling(n^(1/4))
  } else if (type == "c" && !is.null(c)) {
    ceiling(n^(1/4) / c)
  } else {
    stop("Invalid type of entropy")
  }
}

mutualinformation <- function(x_disc,y) {
    H_X <- entropy(x_disc)
    H_Y <- entropy(y)
    H_XY <- infotheo::entropy(cbind(x_disc, y))
    
    # no need for the correction factor as it cancels out
    mi <- H_X + H_Y - H_XY      
}

mutualinformation_btw_features <- function(n,X,x_disc,s_disc) {
  # Calculate entropy error
  # deltas
  delta_univ <- (max(X) - min(X)) / calculate_k(n,"u")
  delta_joint <- (max(X) - min(X)) / calculate_k(n,"j")
  
  H_X <- entropy(x_disc) + log(delta_univ) 
  H_S <- entropy(s_disc) + log(delta_univ) 
  H_SY <- infotheo::entropy(cbind(x_disc, s_disc)) + 2*log(delta_joint)
    
  mi <- H_X + H_S - H_SY
}

# c = length(unique(C))  
mutualinformation_class_relevance <- function(n,c,X,x_disc,s_disc,y) {
  # MI(X_i,X_s|Y) = H(X_i|Y) + H(X_j|Y) - H(X_i,X_j|Y)
  H_y <- entropy(y)
  H_x_y <- entropy(cbind(x_disc, y)) - H_y
  H_s_y <- entropy(cbind(s_disc, y)) - H_y
  H_x_s_y <- entropy(cbind(x_disc, s_disc, y)) - H_y
  
  # Calculate entropy error
  # delta
  delta_join <- (max(X) - min(X)) / calculate_k(n,"u")
  delta_cond <- (max(X) - min(X)) / calculate_k(n,"c",c)
  
  mi <- H_x_y + H_s_y + 2*log(delta_join) - (H_x_s_y + 2*log(delta_cond))
  pmax(0, mi)
}

MIM <- function(X, Y, beta = 0.5, k = ncol(X)) {
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
  
  # Precompute MI(f;Y) - mutual information with target
  mi_target <- sapply(X, function(x) {
    x_disc <- discretize(x)
    mutualinformation(x_disc, Y) / H_Y  # Normalized MI(f;Y)
  })
  
  
  list(
    selection = structure(match(feature_names, feature_names), 
                          names = feature_names),
    score = sort(mi_target, decreasing = TRUE)
  )
}

MIFS <- function(X, Y, beta = 0.5, k = ncol(X)) {
  # Convert input to data frame and factor
  X <- as.data.frame(X)
  Y <- as.factor(Y)
  
  n <- nrow(X)
  
  # Get feature names in original order
  feature_names <- colnames(X)
  
  # Initialize results
  selected <- character(0)
  scores <- setNames(numeric(length(feature_names)), feature_names)
  
  # Precompute entropy
  H_Y <- entropy(Y) 
  
  # Precompute MI(f;Y) - mutual information with target
  mi_target <- sapply(X, function(x) {
    x_disc <- discretize(x)
    mutualinformation(x_disc, Y) / H_Y  # Normalized MI(f;Y)
  })
  
  for (i in 1:k) {
    # Calculate redundancy for remaining features
    remaining <- setdiff(feature_names, selected)
    redundancy <- if (length(selected) > 0) {
      sapply(remaining, function(f) {
        sum(sapply(selected, function(s) {
          # Calculate MI(f;S) - mutual information between features
          mutualinformation_btw_features(n,X,discretize(X[[f]]), discretize(X[[s]])) / H_Y
        }))
      })
    } else {
      setNames(rep(0, length(remaining)), remaining)
    }
    
    # Calculate scores: MI(f;Y) - β*MI(f;S)
    current_scores <- mi_target[remaining] - beta * redundancy
    
    # Select best feature
    best_feature <- names(which.max(current_scores))
    scores[best_feature] <- current_scores[best_feature]
    selected <- c(selected, best_feature)
  }
  
  list(
    selection = structure(match(feature_names, feature_names), 
                          names = feature_names),
    score = sort(scores, decreasing = TRUE)
  )
}

mRMR <- function(X, Y, k = ncol(X)) {
  # Convert input to data frame and factor
  X <- as.data.frame(X)
  Y <- as.factor(Y)
  
  n <- nrow(X)
  
  # Get feature names in original order
  feature_names <- colnames(X)
  
  # Initialize results
  selected <- character(0)
  scores <- setNames(numeric(length(feature_names)), feature_names)
  
  # Precompute entropy
  H_Y <- entropy(Y) 
  
  # Precompute MI(f;Y) - mutual information with target
  mi_target <- sapply(X, function(x) {
    x_disc <- discretize(x)
    mutualinformation(x_disc, Y) / H_Y  # Normalized MI(f;Y)
  })
  
  for (i in 1:k) {
    # Calculate redundancy for remaining features
    remaining <- setdiff(feature_names, selected)
    redundancy <- if (length(selected) > 0) {
      sapply(remaining, function(f) {
        sum(sapply(selected, function(s) {
          # Calculate MI(f;S) - mutual information between features
          mutualinformation_btw_features(n,X,discretize(X[[f]]), discretize(X[[s]])) / H_Y
        })) / length(selected)
      })
    } else {
      setNames(rep(0, length(remaining)), remaining)
    }
    
    # Calculate scores: MI(f;Y) - 1/|S|sum(MI(f;S))
    current_scores <- mi_target[remaining] - redundancy
    
    # Select best feature
    best_feature <- names(which.max(current_scores))
    scores[best_feature] <- current_scores[best_feature]
    selected <- c(selected, best_feature)
  }
  
  list(
    selection = structure(match(feature_names, feature_names), 
                          names = feature_names),
    score = sort(scores, decreasing = TRUE)
  )
}

maxMIFS <- function(X, Y, beta = 1, k = ncol(X)) {
  # Convert input to data frame and factor
  X <- as.data.frame(X)
  Y <- as.factor(Y)
  
  n <- nrow(X)
  
  # Get feature names in original order
  feature_names <- colnames(X)
  
  # Initialize results
  selected <- character(0)
  scores <- setNames(numeric(length(feature_names)), feature_names)
  
  # Precompute entropy
  H_Y <- entropy(Y) 
  
  # Precompute MI(f;Y) - mutual information with target
  mi_target <- sapply(X, function(x) {
    x_disc <- discretize(x)
    mutualinformation(x_disc, Y) / H_Y  # Normalized MI(f;Y)
  })
  
  for (i in 1:k) {
    # Calculate redundancy for remaining features
    remaining <- setdiff(feature_names, selected)
    redundancy <- if (length(selected) > 0) {
      sapply(remaining, function(f) {
        max(sapply(selected, function(s) {
          # Calculate MI(f;S) - mutual information between features
          mutualinformation_btw_features(n,X,discretize(X[[f]]), discretize(X[[s]])) / H_Y
        }))
      })
    } else {
      setNames(rep(0, length(remaining)), remaining)
    }
    
    # Calculate scores: MI(f;Y) - max(MI(f;S))
    current_scores <- mi_target[remaining] - beta * redundancy
    
    # Select best feature
    best_feature <- names(which.max(current_scores))
    scores[best_feature] <- current_scores[best_feature]
    selected <- c(selected, best_feature)
  }
  
  list(
    selection = structure(match(feature_names, feature_names), 
                          names = feature_names),
    score = sort(scores, decreasing = TRUE)
  )
}

CIFE <- function(X, Y, beta = 1, k = ncol(X)) {
  # Convert input keeping native column order
  X <- as.data.frame(X)
  Y <- as.factor(Y)
  
  n <- nrow(X)
  c <- length(unique(Y))  
  
  # Get features in original order
  features <- colnames(X)
  
  # Initialize results
  scores <- setNames(rep(NA, ncol(X)), features)
  selected <- character(0)
  
  # Precompute entropy
  H_Y <- entropy(Y)
  
  for (i in 1:k) {
    remaining <- setdiff(features, selected)
    
    # Calculate CIFE criterion: MI(f;Y) - β[MI(f;S) - CMI(f;S|Y)]
    if (length(selected) > 0) {
      redundancy <- sapply(remaining, function(f) {
        sum(sapply(selected, function(s) {
          # Calculate MI(f;S) - mutual information
          mi_fs <- mutualinformation_btw_features(n,X,discretize(X[[f]]), discretize(X[[s]])) / H_Y
          # Calculate CMI(f;S|Y) - conditional mutual information
          cmi_fs_y <- mutualinformation_class_relevance(n,c,X,discretize(X[[f]]), discretize(X[[s]]), Y) / H_Y
          # Core CIFE term: MI(f;S) - CMI(f;S|Y)
          mi_fs - cmi_fs_y
        }))
      })
    } else {
      redundancy <- setNames(rep(0, length(remaining)), remaining)
    }
    
    # Calculate relevance MI(f;Y)
    relevance <- sapply(remaining, function(f) {
      mutualinformation(discretize(X[[f]]), Y) / H_Y
    })
    
    # CIFE score: MI(f;Y) - β[MI(f;S) - CMI(f;S|Y)]
    current_scores <- relevance - beta * redundancy
    
    # Select best feature without reordering
    best_idx <- which.max(current_scores)
    best_feature <- names(current_scores)[best_idx]
    scores[best_feature] <- current_scores[best_idx]
    selected <- c(selected, best_feature)
  }
  
  # Fill NA with 0 without reordering
  scores[is.na(scores)] <- 0
  
  list(
    selection = setNames(seq_along(features), features),
    score = sort(scores, decreasing = TRUE)
  )
}

JMI <- function(X, Y, beta = 1, k = ncol(X)) {
  # Convert input keeping native column order
  X <- as.data.frame(X)
  Y <- as.factor(Y)
  
  n <- nrow(X)
  c <- length(unique(Y))  
  
  # Get features in original order
  features <- colnames(X)
  
  # Initialize results
  scores <- setNames(rep(NA, ncol(X)), features)
  selected <- character(0)
  
  # Precompute entropy
  H_Y <- entropy(Y)
  
  for (i in 1:k) {
    remaining <- setdiff(features, selected)
    
    # Calculate CIFE criterion: MI(f;Y) - β[MI(f;S) - CMI(f;S|Y)]
    if (length(selected) > 0) {
      redundancy <- sapply(remaining, function(f) {
        sum(sapply(selected, function(s) {
          # Calculate MI(f;S) - mutual information
          mi_fs <- mutualinformation_btw_features(n,X,discretize(X[[f]]), discretize(X[[s]])) / H_Y
          # Calculate CMI(f;S|Y) - conditional mutual information
          cmi_fs_y <- mutualinformation_class_relevance(n,c,X,discretize(X[[f]]), discretize(X[[s]]), Y) / H_Y
          # Core CIFE term: MI(f;S) - CMI(f;S|Y)
          mi_fs - cmi_fs_y
        }))  / length(selected)
      })
    } else {
      redundancy <- setNames(rep(0, length(remaining)), remaining)
    }
    
    # Calculate relevance MI(f;Y)
    relevance <- sapply(remaining, function(f) {
      mutualinformation(discretize(X[[f]]), Y) / H_Y
    })
    
    # CIFE score: MI(f;Y) - β[MI(f;S) - CMI(f;S|Y)]
    current_scores <- relevance - beta * redundancy
    
    # Select best feature without reordering
    best_idx <- which.max(current_scores)
    best_feature <- names(current_scores)[best_idx]
    scores[best_feature] <- current_scores[best_idx]
    selected <- c(selected, best_feature)
  }
  
  # Fill NA with 0 without reordering
  scores[is.na(scores)] <- 0
  
  list(
    selection = setNames(seq_along(features), features),
    score = sort(scores, decreasing = TRUE)
  )
}

CMIM <- function(X, Y, k = ncol(X)) {
  # Convert input to proper formats
  X <- as.data.frame(X)
  Y <- as.factor(Y)
  
  n <- nrow(X)
  c <- length(unique(Y))  
  
  feature_names <- colnames(X)
  
  # Initialize 
  selected <- character(0)
  scores <- setNames(numeric(length(feature_names)), feature_names)
  
  # Precompute entropy
  H_Y <- entropy(Y) 
  
  # Precompute MI(f;Y) - mutual information with target
  mi_target <- sapply(X, function(x) {
    x_disc <- discretize(x)
    mutualinformation(x_disc, Y) / H_Y  # Normalized MI(f;Y)
  })
  
  for (i in 1:k) {
    # Calculate redundancy for remaining features
    remaining <- setdiff(feature_names, selected)
    redundancy <- if (length(selected) > 0) {
      sapply(remaining, function(f) {
        max(sapply(selected, function(s) {
          # Calculate MI(f;S) - mutual information
          mi_fs <- mutualinformation_btw_features(n,X,discretize(X[[f]]), discretize(X[[s]])) / H_Y
          # Calculate CMI(f;Y|S) - conditional mutual information
          cmi_fy_s <- mutualinformation_class_relevance(n,c,X,discretize(X[[f]]), discretize(X[[s]]), Y) / H_Y
          # Core DMIM term: MI(f;S) - CMI(f;Y|S)
          mi_fs - cmi_fy_s
        }))
      })
    } else {
      setNames(rep(0, length(remaining)), remaining)
    }
    
    # CMIM criterion: MI(f;Y) - max[MI(f;S) - CMI(f;Y|S)]
    current_scores <- mi_target[remaining] - redundancy
    
    # Select best feature
    best_feature <- names(which.max(current_scores))
    scores[best_feature] <- current_scores[best_feature]
    selected <- c(selected, best_feature)
  }
  
  list(
    selection = structure(match(feature_names, feature_names), 
                          names = feature_names),
    score = sort(scores, decreasing = TRUE)
  )
}

DMIM <- function(X, Y, k = ncol(X)) {
  # Convert input to proper formats
  X <- as.data.frame(X)
  Y <- as.factor(Y)
  
  n <- nrow(X)
  c <- length(unique(Y))  
  
  feature_names <- colnames(X)
  
  # Initialize 
  selected <- character(0)
  scores <- setNames(numeric(length(feature_names)), feature_names)
  
  # Precompute MI(f;Y) - mutual information with target
  mi_target <- sapply(X, function(x) {
    x_disc <- discretize(x)
    mutualinformation(x_disc, Y) 
  })
  
  for (i in 1:k) {
    # Calculate redundancy for remaining features
    remaining <- setdiff(feature_names, selected)
    
    redundancy <- if (length(selected) > 0) {
      sapply(remaining, function(f) {
        max(sapply(selected, function(s) {
          # Calculate MI(f;S) - mutual information
          mi_fs <- mutualinformation_btw_features(n,X,discretize(X[[f]]), discretize(X[[s]]))
        })) 
        - 
        max(sapply(selected, function(s) {
          # max CMI(f;Y|S) - conditional mutual information
          mutualinformation_class_relevance(n, c, X, discretize(X[[f]]), discretize(X[[s]]), Y) 
        }))
      })
    } else {
      setNames(rep(0, length(remaining)), remaining)
    }
    
    # DMIM criterion: MI(f;Y) - max[MI(f;S)] + max[CMI(f;Y|S)]
    current_scores <- mi_target[remaining] - redundancy
    
    # Select best feature
    best_feature <- names(which.max(current_scores))
    scores[best_feature] <- current_scores[best_feature]
    selected <- c(selected, best_feature)
  }
  
  list(
    selection = structure(match(feature_names, feature_names), 
                          names = feature_names),
    score = sort(scores, decreasing = TRUE)
  )
}

