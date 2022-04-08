# Creating an Artificial Neural Network (ANN) model and explaining it using LIME (Local Interpretable Model-agnostic Explanations)package and Correlation analysis
# Customer Churn is when a customer ends their relationship with a company. As such companies need to focus on reducing customer churn.

# Load libraries
# install.packages("lime")
# install.packages("corrr")

library(tidyverse)
library(keras)
library(lime)   # dis package is used to explain which features drive individual model predictions
library(tidyquant)
library(rsample)   # dis package is for sampling data and generating resamples
library(recipes)   # dis package is for preprocessing ML data sets and performing One-Hot Encoding
library(yardstick)   # dis package is for measuring model metrics/performance
library(corrr)    ## dis package is for Correlation Analysis

mydata <- read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")  # churn column is dependent column. it includes d customers that left d company within d last month
glimpse(mydata)


#---- PreProcess the data ------
# Data wrangling
ourdata <- mydata %>%
  select(-customerID) %>%  # customerID is a unique identifier for each row of value and it is not needed for modeling so we remove it
  drop_na() %>%         # we are removing NA values(becuz it is a very small % of d total dataset) OR u can replace d NAs with value 0
  select(Churn, everything())  # we re-order d columns in d dataset by making d dependent variable/column- Churn to be d 1st variable/column while all d oda columns is after it in d dataset

glimpse(ourdata)


# Split test/training sets using rsample package
set.seed(100)

train <- initial_split(ourdata, prop = 0.8)
train      # displays training, testing and overall total dataset

# Retrieve train and test sets
train_tbl <- training(train)
test_tbl  <- testing(train) 


# Determine if log transformation improves correlation between TotalCharges and Churn
train_tbl %>%
  select(Churn, TotalCharges) %>%
  mutate(Churn = Churn %>% as.factor() %>% as.numeric(),   # changing Churn column to numeric dataype and factor becuz we want to perform correlation (which only works on numerical variables)
         LogTotalCharges = log(TotalCharges)) %>%      # creates a new column- LogTotalCharges with log() values of d existing TotalCharges column as its column values
  correlate() %>%       # performs correlation on numerical columns- Churn and LogTotalCharges columns
  focus(Churn) %>%    # focus() is similar to select()
  fashion()    # makes d formatting easy to read


#------ Preprocessing ---------
# One-Hot Encoding: This is converting d categorical/unique values in a column into sparse data which has columns of only zeros- 0 and ones- 1 (This is also called creating dummy variables)
# so all columns/variable without numerical values will be converted to dummy variables using One-Hot Encoding. This is simple for columns/variables that have only 2 distinct/unique values becuz we can simply convert to 1's and 0's. 
# but it is complicated for columns with more than 2 distinct/unique values as dis requires creating new columns of 1's and 0's for each unique value. We have 4 columns/variables that have more than 2(multiple) distinct/unique values- Contract, Internet Service, Multiple Lines and Payment Method columns
# We will use recipes package for One-Hot Encoding

# ANN models perform much faster and more accurate when the columns/features are scaled/normalized/standardized 
rec_obj <- recipe(Churn ~ ., data = train_tbl) %>%
  step_discretize(tenure, options = list(cuts = 6)) %>%  # Numeric columns like Age, Years worked etc can be generalized into groups/ranges/intervals. tenure column is an example of dis and can be generalized into 6 groups containing diff month ranges/intervals using step_discretize()
  step_log(TotalCharges) %>%     # create log transform of d stated column
  step_dummy(all_nominal(), -all_outcomes()) %>%    # performing one-hot encoding on categorical variables
  step_center(all_predictors(), -all_outcomes()) %>%   # to create mean/center d data. this is feature scale/standardization/normalization of d data
  step_scale(all_predictors(), -all_outcomes()) %>%   # to scale d data. this is feature scale/standardization/normalization of d data
  prep(data = train_tbl)    # preparing d recipe 

rec_obj

# Predictors/independent variables for training and testing sets.
x_train_tbl <- bake(rec_obj, train_tbl) %>% select(-Churn)
glimpse(x_train_tbl)

x_test_tbl  <- bake(rec_obj, test_tbl) %>% select(-Churn)
glimpse(x_test_tbl)


# Response variable for training and testing sets. 
# converting d values in d dependent variable in training dataset to numerical values so its accepted by keras and ANN modeling functions
y_train_vec <- ifelse(pull(train_tbl, Churn) == "Yes", 1, 0)  # means in Churn column in train_tbl object, value- Yes should be converted to 1 else 0(ie value- No will be 0) thus converting d categorical values to numerical values
y_test_vec  <- ifelse(pull(test_tbl, Churn) == "Yes", 1, 0)



# Building our Artificial Neural Network
mymodel <- keras_model_sequential()  # d sequential model is composed of a linear stack of layers

mymodel %>% 
  layer_dense(units = 16, kernel_initializer = "uniform",     # First hidden layer with 16 nodes. input_shape(which is d Input Layer) is number of independent variables/columns in d training set
              activation = "relu", input_shape = ncol(x_train_tbl)) %>% 
  layer_dropout(rate = 0.1) %>%               # Dropout to prevent overfitting
  layer_dense(units = 16, kernel_initializer = "uniform",
              activation = "relu") %>%     # Second hidden layer with 16 nodes.
  layer_dropout(rate = 0.1) %>%      # Dropout layer used to prevent overfitting
  layer_dense(units = 1, kernel_initializer = "uniform", 
              activation = "sigmoid")       # Output layer with 1 node. this is d dependent variable. using sigmoid becuz it is a binary classification

summary(mymodel)


# Compile ANN
mymodel %>% compile(optimizer = 'adam',
                    loss = 'binary_crossentropy',
                    metrics = c('accuracy'))


# Fit the keras model to the training data
myfit <- fit(object = mymodel, 
             x = as.matrix(x_train_tbl), 
             y= y_train_vec, batch_size = 50,   # batch_size is d number samples per gradient update within each epoch
             epochs = 35, validation_split = 0.30)

print(myfit)  # make sure there is only a little diff btw accuracy value and val_accuracy value

# Plot the training/validation history of our Keras model
plot(myfit) 


# creating Prediction using testing data
# mypred <- predict_classes(mymodel, as.matrix(x_test_tbl)) %>% as.vector()   predict_classes() is no longer in use
  
mypred <- mymodel %>% 
  predict(as.matrix(x_test_tbl)) %>%    # we are using dis becus activation = "sigmoid" in d model
  `>`(0.5) %>% 
  k_cast("int32") %>%
  as.numeric()

# mypred1  <- predict_proba(mymodel, as.matrix(x_test_tbl)) %>% as.vector()   predict_probs() is no longer in use
  
mypred1  <- mymodel %>% 
  predict(as.matrix(x_test_tbl)) %>%
  as.numeric()


# using yardstick package to measure/inspect/understand d performance of our model
# using testing data, we create a dataframe/tibble showing d actual values(from testing data), predicted values and probabilities values so as to evaluate d model performance
mymetrics <- tibble(truth = as.factor(y_test_vec) %>%   # truth means Actual values(ie values in testing dataset)
                      fct_recode(yes = "1", no = "0"),  # changes d values of testing data from 1 to yes and from 0 to no
                    estimate = as.factor(mypred) %>%    # estimate means Predicted values
                      fct_recode(yes = "1", no = "0"),  # changes d values of predicted data from 1 to yes and from 0 to no
                    class_prob = mypred1)

mymetrics


options(yardstick.event_first = FALSE)


# Confusion Matrix Table 
mymetrics %>% conf_mat(truth, estimate)  # truth means Actual values, estimate means Predicted values


# Accuracy
mymetrics %>% metrics(truth, estimate)


# ROC Area Under the Curve(AUC)
# AUC is a good metric used to compare diff classifiers and to compare to d randomly guessing AUC value of 0.50
# so if d AUC value is > 0.50 the d model is valid
mymetrics %>% roc_auc(truth, class_prob)  # run dis code X2 to view d correct answers


# Precision; this means when d model predict 'yes', how often is d actual value from d dataset really 'yes'
# Recall; this means when d actual value in d data is 'yes', how often is d model prediction correct
tibble(precision = mymetrics %>%   # all dis is done using yardstick package
         precision(truth, estimate),
       recall = mymetrics %>% 
         recall(truth, estimate))


# F1-Statistic. this is d weighted average btw precision and recall
mymetrics %>% f_meas(truth, estimate, beta = 1)


#----- using LIME package to explain d model ------------ 
class(mymodel)  # copy d 1st row which is keras.engine.sequential.Sequential and apply it to model_type and predict_model below


# creating model_type: here we are telling LIME what type of model are are doing which is classification
model_type.keras.engine.sequential.Sequential <- function(x, ...) {
  return("classification") 
}


# creating predict_model: here we allow LIME to perform predictions that its algorithm can interpret
predict_model.keras.engine.sequential.Sequential <- function(x, newdata, type, ...) {
  pred <- predict(object = x, x = as.matrix(newdata))  
  return(data.frame(Yes = pred, No = 1 - pred))
}


# Test our predict_model() function
?predict_model
predict_model(x = mymodel, newdata = x_test_tbl, 
              type = 'raw') %>% tibble::as_tibble()    # displays d prediction values made on d unique values in Churn column
 

# Run lime() on training set to create an explainer
explain_lime <- lime::lime(x = x_train_tbl, 
                        model = mymodel, 
                        bin_continuous = FALSE)


# Run explain() on explainer
explanation <- lime::explain(x_test_tbl[1:10, ],    # using only 1st 10 rows in d test data becuz if we use d entire test data it will take time to run
                             explainer = explain_lime, 
                             n_labels = 1, n_features = 4,  # using n_labels = 1 becuz we wanto explain only a single class(ie a single uniwue value)
                             kernel_width = 0.5)           # using  n_features = 4 will return d top four features that are critical to each row

# Creating Feature Importance Visualization using explanation object: this shows columns important to d model
# This allows us to visualize d 1st 10 rows of values and also displaying d top four features for each row
plot_features(explanation) +
  labs(title = "LIME Feature Importance Visualization",     
       subtitle = "Hold Out (Test) Set, First 10 Cases Shown")  # The cyan color bars means the feature supports the model conclusion. The red bars contradict


plot_explanations(explanation) +    # this creates a facetted Heatmap of all feature combinations
  labs(title = "LIME Feature Importance Heatmap",
       subtitle = "Hold Out (Test) Set, First 10 Cases Shown")
# To create d Lime visualization above, we only used a sample of d data(we used only d 1st 10 rows)



#----- Checking Explanations with Correlation Analysis using Corr package ----------
# Lets perform Correlation analysis on d training set to see which Features/columns correlates to Churn column
mycor <- x_train_tbl %>%
  mutate(Churn = y_train_vec) %>%  # change d d object name- y_train_vec (which is Response variable for training data) to Churn
  correlate() %>%  # creates a matrix of all d independent numeric variables/columns showing their correlation values 
  focus(Churn) %>%  # shows d correlation values btw each independent numeric variables/columns and Churn column with d independent numeric variables/columns as row values all under a new column- term 
  rename(feature = term) %>%   # rename term column header to feature
  arrange(abs(Churn)) %>%
  mutate(feature = as_factor(feature)) 

mycor


# Correlation Visualization
# This helps to distinguish which features/column are relevant to Churn 
mycor %>%
  ggplot(aes(x = Churn, y = fct_reorder(feature, desc(Churn)))) +
  geom_point() +   # creates dots
  geom_segment(aes(xend = 0, yend = feature),       # Positive Correlations - Contribute to churn
               color = palette_light()[[2]], 
               data = mycor %>% filter(Churn > 0)) +  # creates horizontal lines with d dots at its end based on values > 0 in Churn column
  geom_point(color = palette_light()[[2]], 
             data = mycor %>% filter(Churn > 0)) +
  geom_segment(aes(xend = 0, yend = feature),        # Negative Correlations - Prevent churn
               color = palette_light()[[1]], 
               data = mycor %>% filter(Churn < 0)) +
  geom_point(color = palette_light()[[1]], 
             data = mycor %>% filter(Churn < 0)) +   
  geom_vline(xintercept = 0, color = palette_light()[[5]], size = 1, linetype = 2) +    # creates dashed vertical line at value 0 in x-axis
  geom_vline(xintercept = -0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
  geom_vline(xintercept = 0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
  theme_tq() +                                              # Aesthetics
  labs(title = "Churn Correlation Analysis",
       subtitle = paste("Positive Correlations (contribute to churn),",
                        "Negative Correlations (prevent churn)"),
       y = "Feature Importance")
# Red line means increases likelihood of Churn, Black line decreases likelihood of Churn 


# Feature Investigation: we investigate features/columns that are most frequent in d LIME feature importance Visualization made earlier along with those with a high correlation analysis
mydata %>%
  ggplot(aes(Churn, tenure)) +
  geom_jitter(alpha = 0.25, color = palette_light()[[6]]) +
  geom_violin(alpha = 0.6, fill = palette_light()[[1]]) +
  theme_tq() +
  labs(title = "tenure",
       subtitle = "Customers with lower tenure are more likely to leave")



mydata %>%
  mutate(Churn = ifelse(Churn == "Yes", 1, 0)) %>%  # means if any row value is Yes, input 1, else input 0
  ggplot(aes(as.factor(Contract), Churn)) +
  geom_jitter(alpha = 0.25, color = palette_light()[[6]]) +
  geom_violin(alpha = 0.6, fill = palette_light()[[1]]) +
  theme_tq() +
  labs(title = "Contract Type",
       subtitle = "Two or One Year contracts are much less likely to leave",
       x = "Contract")

       

mydata %>%
  mutate(Churn = ifelse(Churn == "Yes", 1, 0)) %>%
  ggplot(aes(as.factor(InternetService), Churn)) +
  geom_jitter(alpha = 0.25, color = palette_light()[[6]]) +
  geom_violin(alpha = 0.6, fill = palette_light()[[1]]) +
  theme_tq() +
  labs(title = "Internet Service",
       subtitle = "Fiber optic more likely to leave",
       x = "Internet Service")



mydata %>%
  mutate(Churn = ifelse(Churn == "Yes", 1, 0)) %>%
  ggplot(aes(as.factor(PaymentMethod), Churn)) +
  geom_jitter(alpha = 0.25, color = palette_light()[[6]]) +
  geom_violin(alpha = 0.6, fill = palette_light()[[1]]) +
  theme_tq() +
  labs(title = "Payment Method",
       subtitle = "Electronic check more likely to leave",
       x = "Payment Method")



mydata %>%
  mutate(Churn = ifelse(Churn == "Yes", 1, 0)) %>%
  ggplot(aes(as.factor(SeniorCitizen), Churn)) +
  geom_jitter(alpha = 0.25, color = palette_light()[[6]]) +
  geom_violin(alpha = 0.6, fill = palette_light()[[1]]) +
  theme_tq() +
  labs(title = "Senior Citizen",
       subtitle = "Non-senior citizens less likely to leave",
       x = "Senior Citizen")



mydata %>%
  mutate(Churn = ifelse(Churn == "Yes", 1, 0)) %>%
  ggplot(aes(OnlineSecurity, Churn)) +
  geom_jitter(alpha = 0.25, color = palette_light()[[6]]) +
  geom_violin(alpha = 0.6, fill = palette_light()[[1]]) +
  theme_tq() +
  labs(title = "Online Security",
       subtitle = "Customers without online security are more likely to leave")
      