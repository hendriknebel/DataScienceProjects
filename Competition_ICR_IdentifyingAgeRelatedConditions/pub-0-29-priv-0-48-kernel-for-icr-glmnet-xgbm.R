# Title: Pub 0.29 Priv 0.48 Kernel for ICR (GLMNET, XGBM)
# Author: Hendrik Nebel
# Date: August 20, 2023

### Libraries ---------------------------------------------------------
library(data.table)
library(skimr)
library(plyr)
library(caret)
library(caretEnsemble)
library(xgboost)
library(kernlab)
library(Matrix)
library(recipes)
library(randomForest)
# ---------------------------------------------------------------------

### Data Loading ------------------------------------------------------
data_train_loaded <- fread("../input/icr-identify-age-related-conditions/train.csv")
data_test_loaded <- fread("../input/icr-identify-age-related-conditions/test.csv")
data_test_loaded[["Class"]] <- rep(NA, nrow(data_test_loaded))
data_entire <- rbind(data_train_loaded, data_test_loaded)
remove(data_train_loaded); remove(data_test_loaded)
# ---------------------------------------------------------------------

### Data Preparation --------------------------------------------------
data_entire$Class <- as.factor(data_entire$Class)                       # convert response as factor
levels(data_entire$Class) <- c("class_0", "class_1")                    # rename levels - "0", "1" can cause errors
vec_type <- sapply(data_entire, typeof)                                 # vector for types

# Select Factor Variables
col.fac <- names(vec_type)[vec_type == "character"]
col.fac <- c(col.fac, "EJ")

# Select Numeric Variables
col.num <- names(vec_type)[vec_type == "integer"]

# Convert Characters -> Factors
for(col in col.fac){
    data_entire[[col]] <- factor(data_entire[[col]])
}

options("max.print" = 100000)                                           # limit output

skim_with(integer = list(complete = NULL,
                         n = NULL,
                         sd = NULL),
          factor = list(ordered = NULL))
skim(data_entire)
# ---------------------------------------------------------------------

### Using Recipe ------------------------------------------------------
data_train_select <- data_entire[!is.na(Class)]
data_test_select <- data_entire[is.na(Class)]
set.seed(1)
rec_obj <- recipe(Class ~ ., data = data_train_select) %>%
    update_role(Id, new_role = "id var") %>%
    step_impute_knn(all_predictors()) %>%
    step_dummy(all_predictors(), -all_numeric()) %>%
    step_YeoJohnson(all_predictors()) %>%
    step_center(all_predictors()) %>%
    step_scale(all_predictors()) %>%
    step_zv(all_predictors()) %>%
    step_corr(all_predictors(), threshold = .9) %>%
    check_missing(all_predictors())

rec_obj
trained_rec <- prep(rec_obj, training = data_train_select)
data_train <- bake(trained_rec, new_data = data_train_select)
data_test <- bake(trained_rec, new_data = data_test_select)
# ---------------------------------------------------------------------

### Define Balanced Log Loss Metric -----------------------------------
# balanced_log_loss function
balanced_log_loss <- function(y_true, y_pred, eps = 1e-15) {
    N0 <- sum(y_true == 0)
    N1 <- sum(y_true == 1)
    p1 <- pmax(pmin(y_pred, 1-eps), eps)
    p0 <- 1 - p1
    result <- (sum((1-y_true) * log(p0)) / N0 + sum(y_true * log(p1)) / N1) / -2
    return(result)
}

# balanced_log_loss_summary function - used for the caret package
balanced_log_loss_summary <- function(data, lev, model) {
  # Convert levels back to "0" & "1"
  levels(data$obs) <- c(0, 1)
  levels(data$pred) <- c(0, 1)
  # Extract predicted class probabilities or predictions
  predictions <- as.numeric(as.character(data$class_1))
  
  # Extract the actual outcomes
  actual <- as.numeric(as.character(data$obs))

  blogloss <- balanced_log_loss(actual, predictions)

  c(blogloss = blogloss)
}

####
# IDEA: Instead of simply using the given Evaluation Metric, use a stricter version to search for models that perform under the Eval Metric EVEN IF the computed predicted probabilities get set to a perfect prediction value (either "0" or "1" when crossing a threshold)
#
# The following Eval Metric (_HARD) function is used ONLY to find models under harder conditions.
# The Original Eval Metric is later used to provide the CV performance with the found models.
###
# balanced_log_loss_summary_HARD function - used for the caret package
balanced_log_loss_summary_HARD <- function(data, lev, model){
  # Convert levels back to "0" & "1"
  levels(data$obs) <- c(0, 1)
  levels(data$pred) <- c(0, 1)
  # Extract predicted class probabilities or predictions
  predictions <- as.numeric(as.character(data$class_1))

  # Create thresholds at which predictions are set to either 0 or 1
  empty <- numeric(length = length(predictions))
  for(i in 1:length(predictions)){
      #print(predictions[i])
      if((!is.na(predictions[i])) && (!is.infinite(predictions[i])) && (predictions[i] > 0.86)){
          empty[i] <- 1 
      } else if((!is.na(predictions[i])) && (!is.infinite(predictions[i])) && (predictions[i] < 0.14)){
          empty[i] <- 0
      } else{
          empty[i] <- predictions[i]
      }
  }
  predictions <- empty
  
  # Extract the actual outcomes
  actual <- as.numeric(as.character(data$obs))

  blogloss <- balanced_log_loss(actual, predictions)

  c(blogloss = blogloss)
}
# ---------------------------------------------------------------------



### Predictive Modeling -----------------------------------------------
### Define my_control -------------------------------------------------
my_control <- trainControl(method = "cv",                                       # Cross-Validation
                           number = 10,                                         # 10 folds
                           summaryFunction = balanced_log_loss_summary_HARD,    # Eval Metric (HARD)
                           classProb = T,                                       # get Class Probabilities
                           sampling = "up")                                     # Oversampling the Minority Class "1"
# ---------------------------------------------------------------------
# ### GLMNET Modeling .................................................
# tune_grid <- expand.grid(alpha = seq(0.1, 1, 0.1),
#                          lambda = seq(0.1, 1, by = 0.1))

# tune_grid <- expand.grid(alpha = seq(0.05, 0.2, 0.01),
#                          lambda = 0.2)


# tune_grid <- expand.grid(alpha = 0.18,
#                          lambda = 0.24)

# # Best
# # alpha = 0.1; lambda = 0.2
# # 0.18; 0.24
# # 0.12; 0.2
# # 0.12; 0.24

# glmnet_cv <- train(Class ~ .,
#                       data = data_train,
#                       method = "glmnet",
#                       family = "binomial",
#                       metric = "blogloss",
#                       maximize = F,
#                       trControl = my_control,
#                       tuneGrid = tune_grid)

# print(glmnet_cv)
# print(glmnet_cv$resample)
# message("CV Mean: ", round(mean(glmnet_cv$resample[[1]]), digits = 4))
# message("CV SD: ", round(sd(glmnet_cv$resample[[1]]), digits = 4))
# # ...................................................................
# ### XGBoost Modeling ................................................
# tune_grid <- expand.grid(nrounds = 200,
#                          max_depth = 90,
#                          eta = 0.016,
#                          gamma = 6.5,
#                          colsample_bytree = 0.14,
#                          min_child_weight = 7,
#                          subsample = 0.66)

# set.seed(1)
# xgbm_cv <- train(Class ~ .,
#                     data = data_train_04,
#                     method = "xgbTree",
#                     metric = "blogloss",
#                     maximize = F,
#                     trControl = my_control,
#                     tuneGrid = tune_grid)

# print(xgbm_cv)
# print(xgbm_cv$resample)
# message("CV Mean: ", round(mean(xgbm_cv$resample[[1]]), digits = 4))
# message("CV SD: ", round(sd(xgbm_cv$resample[[1]]), digits = 4))
# # ...................................................................
# ---------------------------------------------------------------------



### Ensembling --------------------------------------------------------
## Define my_ens_control ..............................................
my_ens_control <- trainControl(method = "cv",                                                       # Cross-Validation
                               savePredictions = "final",
                               index = createFolds(data_train$Class, k = 10, returnTrain = TRUE),   # use same folds
                               allowParallel = TRUE,
                               verboseIter = TRUE,
                               classProb = TRUE,
                               sampling = "up",                                                     # Oversample Minority Class "1"
                               summaryFunction = balanced_log_loss_summary)                         # Eval Metric
# .....................................................................
# Define Grids ........................................................
glmnetGrid <- expand.grid(alpha = 0.18, lambda = 0.24)

xgbTreeGrid <- expand.grid(nrounds = 200,
                           max_depth = 25,
                           eta = 0.1,
                           gamma = 2,
                           colsample_bytree = 0.4,
                           subsample = 0.63,
                           min_child_weight = 4) 

### Define tune_list ..................................................
tune_list <- list(GLMNET = caretModelSpec(method = "glmnet", tuneGrid = glmnetGrid),
                  XGBM = caretModelSpec(method = "xgbTree", tuneGrid = xgbTreeGrid))
# .....................................................................
# Computing the Models ................................................
set.seed(1)
modelList <- caretList(x = subset(data_train, select = -c(Id, Class)),
                       y = data_train$Class,
                       trControl = my_ens_control,
                       metric = "blogloss",
                       tuneList = tune_list)
# .....................................................................
# ---------------------------------------------------------------------

### Stacking Model ----------------------------------------------------
set.seed(1)
greedyEnsemble <- caretEnsemble(modelList,                                                              # combine Models
                                metric  = "blogloss",
                                trControl = trainControl(method = "repeatedcv",                         # Repeated Cross-Validation
                                                         number = 10,
                                                         repeats = 3,
                                                         summaryFunction = balanced_log_loss_summary))  # Eval Metric
summary(greedyEnsemble)

# data_preds_train <- data.frame(GLMNET = predict(greedyEnsemble$models$GLMNET, subset(data_train, select = -c(Id, Class)), type = "prob")[[1]],
#                                XGBM = predict(greedyEnsemble$models$XGBM, subset(data_train, select = -c(Id, Class)), type = "prob")[[1]],
#                                Class = data_train$Class)

# greedyCheck <- train(Class ~ GLMNET + XGBM,
#                      data = data_preds_train,
#                      method = "glm",
#                      family = "binomial",
#                      trControl = trainControl(method = "cv", number = 10, summaryFunction = balanced_log_loss_summary, classProb = TRUE),
#                      metric = "blogloss")

# round(mean(greedyCheck$resample[[1]]), digits = 4)
# round(sd(greedyCheck$resample[[1]]), digits = 4)

# preds <- predict(greedyEnsemble, newdata = subset(data_train, select = -c(Id, Class)), type = "prob")
# preds_02 <- predict(greedyCheck, newdata = subset(data_preds_train, select = -c(Class)), type = "prob")[[1]]

# vec_empty <- data_train$Class
# levels(vec_empty) <- c("1", "0")
# vec_empty <- as.numeric(as.character(vec_empty))

# balanced_log_loss(vec_empty, preds)
# balanced_log_loss(vec_empty, preds_02)
# ---------------------------------------------------------------------



### Submission --------------------------------------------------------
data_preds_target <- predict(greedyEnsemble,
                             newdata = subset(data_test, select = -c(Id, Class)),
                             type = "prob")

data_submission <- data.frame("Id" = data_entire[is.na(Class), 1],
                              "class_0" = data_preds_target,
                              "class_1" = 1 - data_preds_target)

write.csv(data_submission,
          file = "submission.csv",
          row.names = F)
# ---------------------------------------------------------------------

# -------- THANKS FOR READING! Comments & Recommendations appreciated! --------
# ----- Don't forget to upvote, if you like the kernel. -----
