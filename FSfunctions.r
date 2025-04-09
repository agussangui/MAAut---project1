library(infotheo)

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

mutualinformation_btw_features <- function(n,x,s) {
  x_disc <- discretize(x)
  s_disc <- discretize(s)
  
  # Calculate entropy error
  # deltas
  delta_univ_x <- (max(x) - min(x)) / calculate_k(n,"u")
  delta_univ_s <- (max(s) - min(s)) / calculate_k(n,"u")
  delta_joint_x <- (max(x) - min(x)) / calculate_k(n,"j")
  delta_joint_s <- (max(s) - min(s)) / calculate_k(n,"j")
  
  H_X <- entropy(x_disc) + log(delta_univ_x) 
  H_S <- entropy(s_disc) + log(delta_univ_s) 
  H_XS <- infotheo::entropy(cbind(x_disc, s_disc)) + log(delta_joint_x) + log(delta_joint_s)
    
  mi <- H_X + H_S - H_XS
}

# c = length(unique(C))  
mutualinformation_class_relevance <- function(n,c,x,s,y) {
  x_disc <- discretize(x)
  s_disc <- discretize(s)
  
  # Calculate entropy error
  # delta
  delta_univ_x <- (max(x) - min(x)) / calculate_k(n,"u")
  delta_univ_s <- (max(s) - min(s)) / calculate_k(n,"u")
  
  delta_cond_x <- (max(x) - min(x)) / calculate_k(n,"c",c)
  delta_cond_s <- (max(s) - min(s)) / calculate_k(n,"c",c)
  
  # MI(X_i,X_s|Y) = H(X_i|Y) + H(X_j|Y) - H(X_i,X_j|Y)
  H_Y <- entropy(y)
  H_XY <- entropy(cbind(x_disc, y)) - H_Y + log(delta_univ_x) 
  H_SY <- entropy(cbind(s_disc, y)) - H_Y + log(delta_univ_s) 
  H_XSY <- entropy(cbind(x_disc, s_disc, y)) - H_Y + log(delta_cond_x) + log(delta_cond_s) 
  
  mi <- H_XY + H_SY - H_XSY
  pmax(0, mi)
}

MIM <- function(X, Y, k = ncol(X)) {
  # Convert input to data frame and factor
  X <- as.data.frame(X)
  Y <- as.factor(Y)
  
  # Get feature names in original order
  feature_names <- colnames(X)
  
  # Initialize results
  selected <- character(0)
  scores <- setNames(numeric(length(feature_names)), feature_names)
  
  # Precompute MI(f;Y) - mutual information with target
  mi_target <- sapply(X, function(x) {
    x_disc <- discretize(x)
    mutualinformation(x_disc, Y)
  })
  
  
  ordered_features <- names(sort(mi_target, decreasing = TRUE))
  
  list(
    selection = structure(match(ordered_features, feature_names), names = ordered_features),
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
        sum(sapply(selected, function(s) {
          # Calculate MI(f;S) - mutual information between features
          mutualinformation_btw_features(n,X[[f]], X[[s]])
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
  
  ordered_features <- names(sort(mi_target, decreasing = TRUE))
  
  list(
    selection = structure(match(ordered_features, feature_names), names = ordered_features),
    score = sort(mi_target, decreasing = TRUE)
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
        sum(sapply(selected, function(s) {
          # Calculate MI(f;S) - mutual information between features
          mutualinformation_btw_features(n,X[[f]], X[[s]] )
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
  
  ordered_features <- names(sort(mi_target, decreasing = TRUE))
  
  list(
    selection = structure(match(ordered_features, feature_names), names = ordered_features),
    score = sort(mi_target, decreasing = TRUE)
  )
}

maxMIFS <- function(X, Y, k = ncol(X)) {
  # Convert input to data frame and factor
  X <- as.data.frame(X)
  Y <- as.factor(Y)
  
  n <- nrow(X)
  
  # Get feature names in original order
  feature_names <- colnames(X)
  
  # Initialize results
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
          # Calculate MI(f;S) - mutual information between features
          mutualinformation_btw_features(n,X[[f]], X[[s]] )
        }))
      })
    } else {
      setNames(rep(0, length(remaining)), remaining)
    }
    
    # Calculate scores: MI(f;Y) - max(MI(f;S))
    current_scores <- mi_target[remaining] - redundancy
    
    # Select best feature
    best_feature <- names(which.max(current_scores))
    scores[best_feature] <- current_scores[best_feature]
    selected <- c(selected, best_feature)
  }
  
  ordered_features <- names(sort(mi_target, decreasing = TRUE))
  
  list(
    selection = structure(match(ordered_features, feature_names), names = ordered_features),
    score = sort(mi_target, decreasing = TRUE)
  )
}

CIFE <- function(X, Y, k = ncol(X)) {
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
  
  for (i in 1:k) {
    remaining <- setdiff(features, selected)
    
    # Calculate CIFE criterion: MI(f;Y) - β[MI(f;S) - CMI(f;S|Y)]
    if (length(selected) > 0) {
      redundancy <- sapply(remaining, function(f) {
        sum(sapply(selected, function(s) {
          # Calculate MI(f;S) - mutual information
          mi_fs <- mutualinformation_btw_features(n,X[[f]], X[[s]] )
          # Calculate CMI(f;S|Y) - conditional mutual information
          cmi_fs_y <- mutualinformation_class_relevance(n,c,X[[f]],X[[s]], Y)
          # Core CIFE term: MI(f;S) - CMI(f;S|Y)
          mi_fs - cmi_fs_y
        }))
      })
    } else {
      redundancy <- setNames(rep(0, length(remaining)), remaining)
    }
    
    # Calculate relevance MI(f;Y)
    relevance <- sapply(remaining, function(f) {
      mutualinformation(discretize(X[[f]]), Y) 
    })
    
    # CIFE score: MI(f;Y) - sum[MI(f;S) - CMI(f;S|Y)]
    current_scores <- relevance - redundancy
    
    # Select best feature without reordering
    best_idx <- which.max(current_scores)
    best_feature <- names(current_scores)[best_idx]
    scores[best_feature] <- current_scores[best_idx]
    selected <- c(selected, best_feature)
  }
  
  # Fill NA with 0 without reordering
  scores[is.na(scores)] <- 0
  
  ordered_features <- names(sort(scores, decreasing = TRUE))
  
  list(
    selection = structure(match(ordered_features, features), names = ordered_features),
    score = sort(scores, decreasing = TRUE)
  )
}

JMI <- function(X, Y, k = ncol(X)) {
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
  
  for (i in 1:k) {
    remaining <- setdiff(features, selected)
    
    # Calculate JMI criterion: MI(f;Y) - 1/|S|sum[MI(f;S) - CMI(f;S|Y)]
    if (length(selected) > 0) {
      redundancy <- sapply(remaining, function(f) {
        sum(sapply(selected, function(s) {
          # Calculate MI(f;S) - mutual information
          mi_fs <- mutualinformation_btw_features(n,X[[f]], X[[s]] )
          # Calculate CMI(f;S|Y) - conditional mutual information
          cmi_fs_y <- mutualinformation_class_relevance(n,c,X[[f]],X[[s]], Y)
          # Core JMI term: MI(f;S) - CMI(f;S|Y)
          mi_fs - cmi_fs_y
        }))  / length(selected)
      })
    } else {
      redundancy <- setNames(rep(0, length(remaining)), remaining)
    }
    
    # Calculate relevance MI(f;Y)
    relevance <- sapply(remaining, function(f) {
      mutualinformation(discretize(X[[f]]), Y)
    })
    
    # JMI score: MI(f;Y) - 1/|S|sum[MI(f;S) - CMI(f;S|Y)]
    current_scores <- relevance - redundancy
    
    # Select best feature without reordering
    best_idx <- which.max(current_scores)
    best_feature <- names(current_scores)[best_idx]
    scores[best_feature] <- current_scores[best_idx]
    selected <- c(selected, best_feature)
  }
  
  # Fill NA with 0 without reordering
  scores[is.na(scores)] <- 0
  
  ordered_features <- names(sort(scores, decreasing = TRUE))
  
  list(
    selection = structure(match(ordered_features, features), names = ordered_features),
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
          mi_fs <- mutualinformation_btw_features(n,X[[f]], X[[s]] )
          # Calculate CMI(f;Y|S) - conditional mutual information
          cmi_fy_s <- mutualinformation_class_relevance(n,c,X[[f]],X[[s]], Y)
          # Core CMIM term: MI(f;S) - CMI(f;Y|S)
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
  
  ordered_features <- names(sort(scores, decreasing = TRUE))
  
  list(
    selection = structure(match(ordered_features, feature_names), names = ordered_features),
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
          mi_fs <- mutualinformation_btw_features(n,X[[f]], X[[s]] )
        })) 
        - 
        max(sapply(selected, function(s) {
          # max CMI(f;Y|S) - conditional mutual information
          mutualinformation_class_relevance(n,c,X[[f]],X[[s]], Y)
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
  
  ordered_features <- names(sort(scores, decreasing = TRUE))
  
  list(
    selection = structure(match(ordered_features, feature_names), names = ordered_features),
    score = sort(scores, decreasing = TRUE)
  )
}

