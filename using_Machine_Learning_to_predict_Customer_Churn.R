# using Machine Learning algorithms: Logistic Regression, Decision Trees and Random Forest to predict Customer Churn

# Customer churn or Customer attrition occurs when customers/subscribers stop doing business with a company or service. It is also referred to as loss of clients/customers

library(plyr)
library(dplyr)
library(corrplot)
library(ggplot2)
library(gridExtra)
library(ggthemes)
library(caret)
library(MASS)
library(randomForest)
library(party)

mychurn <- read.csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
str(mychurn)

# change any column/variable that looks like it should be a categorical/factor variable to factor datatype
mychurn <- mychurn |> mutate(across(c(customerID, gender, Partner, Dependents, 
                                      PhoneService:PaymentMethod, Churn), as.factor))
str(mychurn) 

head(mychurn)
dim(mychurn)


#----- Checking for Missing Values ------
# lets check d number of missing values in each column using sapply() and a function argument
sapply(mychurn, function(x) sum(is.na(x)))

# removing all rows with missing values
mychurn <- mychurn[complete.cases(mychurn),]  

# checking again for missing values
sapply(mychurn, function(x) sum(is.na(x)))

dim(mychurn)


# ------ Data Wrangling ------
# lets see total number of values for each unique value in OnlineSecurity column
unique(mychurn['OnlineSecurity'])  # let us change value- No internet service to just value- No so that there are only 2 unique values


# lets change d value- No internet service to value- No in columns 10 to 15
cols_recode1 <- c(10:15)  # dis are d columns that contain value- No internet service
for (i in 1:ncol(mychurn[, cols_recode1])) {
  mychurn[, cols_recode1][, i] <- as.factor(mapvalues(mychurn[, cols_recode1][, i],     # converts d columns 10 to 15 to factor datatype
                                                      from = c("No internet service"), to = c("No")))  # so we change No internet service to No
}

# lets also change MultipleLines column to a factor and d value- No phone service to value- No in MultipleLines column using mapvalues()
mychurn$MultipleLines <- as.factor(mapvalues(mychurn$MultipleLines, 
                                             from = c("No phone service"), to = c("No")))

# displays d min and max values of tenure column(values are in months)
min(mychurn$tenure); max(mychurn$tenure)

# since d min is 1 month and max is 72 months in tenure column, lets create 5 groups containing diff month ranges/intervals using function argument
group_tenure <- function(tenure){
  if(tenure >= 0 & tenure <= 12){
    return('0 - 12 Month')         # 1st group- 0 to 12 months
  }else if(tenure > 12 & tenure <= 24){
    return('12 - 24 Month')         # 2st group- 12 to 24 months
  }else if(tenure > 24 & tenure <= 48){
    return('24 - 48 Month')         # 3rd group- 24 to 48 months
  }else if(tenure > 48 & tenure <= 60){
    return('48 - 60 Month')        # 4th group- 48 to 60 months
  }else if(tenure > 60){
    return('> 60 Month')            # 5th group- > 60 months
  }
}

mychurn$tenure_group <- sapply(mychurn$tenure, group_tenure)  # replaces/changes each value in tenure column to its corresponding ranges/interval made earlier, all in a new column
head(mychurn$tenure_group, 10)

mychurn$tenure_group <- as.factor(mychurn$tenure_group)  # changes dis new tenure_group column to a factor datatype

str(mychurn)

# Changes SeniorCitizen column to factor datatype and changes d existing values to d stated values using mapvalues()
mychurn$SeniorCitizen <- as.factor(mapvalues(mychurn$SeniorCitizen, from = c("0", "1"), to = c("No", "Yes")))

str(mychurn)

# let remove d columns below from d dataset becuz we do not need dem
mychurn$customerID <- NULL
mychurn$tenure <- NULL

str(mychurn)


#----- Exploratory Data Analysis and Feature Selection -----------
# lets check for Correlation btw d numerical variables/columns in d dataset
numeric_var <- sapply(mychurn, is.numeric)  # checking to see which columns in d dataset is numeric
numeric_var 

corr_matrix <- cor(mychurn[, numeric_var])  # displays a matrix of d numerical columns and their correlation values
corr_matrix

corrplot(corr_matrix, main = "\n\nCorrelation Plot for Numerical Variables", method = "number") # displays d matrix plot of d numerical columns and their correlation values 
# we can see from d matrix and matrix plot that d 2 columns are correlated(ie have identical values) so we have to remove one of them from d model 

# removes TotalCharges column from d dataset
mychurn$TotalCharges <- NULL
str(mychurn)


#------ Creating Visualizations -----------
# creating bar charts for each categorical variable/column(all columns that do not contain numerical values)
p1 <- ggplot(mychurn, aes(x = gender)) +  # using Gender column and its frequency(count)
  ggtitle("Gender") + xlab("Gender") +
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) +   
  ylab("Percentage") +
  coord_flip() +
  theme_minimal()
p1


p2 <- ggplot(mychurn, aes(x = SeniorCitizen)) +  # using SeniorCitizen column and its frequency(count)
  ggtitle("Senior Citizen") + 
  xlab("Senior Citizen") +
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) +
  ylab("Percentage") +
  coord_flip() +
  theme_minimal()
p2


p3 <- ggplot(mychurn, aes(x = Partner)) +  # using Partner column and its frequency(count)
  ggtitle("Partner") +
  xlab("Partner") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + 
  ylab("Percentage") +
  coord_flip() +
  theme_minimal()
p3


p4 <- ggplot(mychurn, aes(x = Dependents)) +  # using Dependents column and its frequency(count)
  ggtitle("Dependents") +
  xlab("Dependents") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + 
  ylab("Percentage") +
  coord_flip() +
  theme_minimal()
p4


grid.arrange(p1, p2, p3, p4, ncol=2)  # displays d stated charts objects all together in one layout in a 2 column format


p5 <- ggplot(mychurn, aes(x = PhoneService)) +  # using PhoneService column and its frequency(count)
  ggtitle("Phone Service") + 
  xlab("Phone Service") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + 
  ylab("Percentage") +
  coord_flip() +
  theme_minimal()
p5


p6 <- ggplot(mychurn, aes(x = MultipleLines)) +  # using MultipleLines column and its frequency(count)
  ggtitle("Multiple Lines") + 
  xlab("Multiple Lines") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + 
  ylab("Percentage") +
  coord_flip() +
  theme_minimal()
p6


p7 <- ggplot(mychurn, aes(x = InternetService)) +  # using InternetService column and its frequency(count)
  ggtitle("Internet Service") + 
  xlab("Internet Service") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + 
  ylab("Percentage") +
  coord_flip() +
  theme_minimal()
p7


p8 <- ggplot(mychurn, aes(x = OnlineSecurity)) +   # using OnlineSecurity column and its frequency(count)
  ggtitle("Online Security") + 
  xlab("Online Security") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + 
  ylab("Percentage") +
  coord_flip() +
  theme_minimal()
p8


grid.arrange(p5, p6, p7, p8, ncol=2)  # displays d stated charts objects all together in one layout in a 2 column format


p9 <- ggplot(mychurn, aes( x = OnlineBackup)) +  # using OnlineBackup column and its frequency(count)
  ggtitle("Online Backup") + 
  xlab("Online Backup") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + 
  ylab("Percentage") +
  coord_flip() +
  theme_minimal()
p9


p10 <- ggplot(mychurn, aes(x = DeviceProtection)) +  # using DeviceProtection column and its frequency(count)
  ggtitle("Device Protection") + 
  xlab("Device Protection") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + 
  ylab("Percentage") + 
  coord_flip() + 
  theme_minimal()
p10


p11 <- ggplot(mychurn, aes(x = TechSupport)) +  # using TechSupport column and its frequency(count)
  ggtitle("Tech Support") + 
  xlab("Tech Support") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + 
  ylab("Percentage") +
  coord_flip() +
  theme_minimal()
p11


p12 <- ggplot(mychurn, aes(x = StreamingTV)) +  # using StreamingTV column and its frequency(count)
  ggtitle("Streaming TV") + 
  xlab("Streaming TV") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + 
  ylab("Percentage") +
  coord_flip() +
  theme_minimal()
p12


grid.arrange(p9, p10, p11, p12, ncol=2)  # displays d stated charts objects all together in one layout in a 2 column format


p13 <- ggplot(mychurn, aes(x = StreamingMovies)) +  # using StreamingMovies column and its frequency(count)
  ggtitle("Streaming Movies") + 
  xlab("Streaming Movies") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + 
  ylab("Percentage") +
  coord_flip() +
  theme_minimal()
p13


p14 <- ggplot(mychurn, aes(x = Contract)) +  # using Contract column and its frequency(count)
  ggtitle("Contract") + 
  xlab("Contract") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + 
  ylab("Percentage") +
  coord_flip() + 
  theme_minimal()
p14


p15 <- ggplot(mychurn, aes(x = PaperlessBilling)) +  # using PaperlessBilling column and its frequency(count)
  ggtitle("Paperless Billing") + 
  xlab("Paperless Billing") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + 
  ylab("Percentage") +
  coord_flip() + 
  theme_minimal()
p15


p16 <- ggplot(mychurn, aes(x = PaymentMethod)) +  # using PaymentMethod column and its frequency(count)
  ggtitle("Payment Method") + 
  xlab("Payment Method") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + 
  ylab("Percentage") +
  coord_flip() +
  theme_minimal()
p16


p17 <- ggplot(mychurn, aes(x = tenure_group)) +  # using tenure_group column and its frequency(count)
  ggtitle("Tenure Group") + 
  xlab("Tenure Group") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + 
  ylab("Percentage") +
  coord_flip() + 
  theme_minimal()
p17


grid.arrange(p13, p14, p15, p16, p17, ncol=2)  # displays d stated charts objects all together in one layout in a 2 column format



#----- Logistical Regression Model -----------
# Partition Churn column(d dependent variable) in d dataset
intrain <- createDataPartition(mychurn$Churn, 
                               p = 0.7, list = FALSE)

set.seed(2018)

training <- mychurn[intrain, ]
testing <- mychurn[- intrain, ]

dim(training); dim(testing)   # displays number of rows and column in training data and testing data to confirm d partitioning

# creating Logistical Regression model
LogModel <- glm(Churn ~ ., family = binomial(link = "logit"),   # Churn column is dependent variable
                data = training)  
                
print(summary(LogModel))  # d columns with 3 stars(***) are d most important columns to d model

anova(LogModel, test = "Chisq")  # columns with 3 stars(***) in Pr(>Chi) column which is p-values are significant columns


# Prediction on testing data
# lets change d columns to character datatype to be same as testing data. we are doing dis so that d confusion matrix will have values 0 and 1 as column and row headers for Actual and Predicted  
testing$Churn <- as.character(testing$Churn)  # changing d datatype of Churn column in testing data
testing$Churn[testing$Churn == "No"] <- "0"   # changing d existing value- No to value- 0 in Churn column
testing$Churn[testing$Churn == "Yes"] <- "1"   # changing d existing value- Yes to value- 1 in Churn column
str(testing)

fitted_results <- predict(LogModel, newdata = testing, type = "response")
head(fitted_results)

fitted_results <- ifelse(fitted_results > 0.5, 1, 0)

# Misclassification Error
misClasificError <- mean(fitted_results != testing$Churn)
misClasificError

# Accuracy
print(paste('Logistic Regression Accuracy is: ', 1- misClasificError))

# Confusion Matrix
print("Confusion Matrix for Logistic Regression");

table(testing$Churn, fitted_results > 0.5)  


# Odds Ratio: Odds ratio is what are the odds dat an event will happen.
library(MASS)

exp(cbind(OR = coef(LogModel), confint(LogModel)))  # OR is Odds Ratio




#------- Decision Tree Model -----------
# We are using only d 3 columns (dat we found out was d most significant to d Logistical Regression model earlier) in dis Decision Tree model
mytree <- ctree(Churn ~ Contract + tenure_group + PaperlessBilling,  # We are using only d 3 columns (dat we found out was d most significant to d Logistical Regression model earlier) in dis Decision Tree model
                training)  
plot(mytree)    # displays decision tree diagram. The 1st/top column in d diagram is d most important column to d model since it determines which decision will occur(ie Contract column as d 1st/top column is most important to d model)
# as such Contract column is d most important column to be used to predict Churn column from d dataset
# from d decision tree diagram; If a customer has a one-year or two-year contract, no matter if he (she) has PapelessBilling or not, he (she) is less likely to churn(since No probability is much higher than Yes in d stacked bar chart) .
# if a customer has a month-to-month contract, and is in the tenure group of 0 - 12 month, and using PaperlessBilling, then this customer is more likely to churn(since No probability is not dat higher than Yes in d stacked bar chart) ..


# Prediction on testing data
pred_tree <- predict(mytree, testing)

# Confusion Matrix
print("Confusion Matrix for Decision Tree"); table(Predicted = pred_tree, Actual = testing$Churn)

# Prediction on training data
p1 <- predict(mytree, training)

# Confusion Matrix of training data
tab1 <- table(Predicted = p1, Actual = training$Churn)

# Confusion Matrix of testing data
tab2 <- table(Predicted = pred_tree, Actual = testing$Churn)

# Accuracy
print(paste('Decision Tree Accuracy',sum(diag(tab2))/sum(tab2)))





#----------- Random Forest Model --------------
rfModel <- randomForest(Churn ~., data = training)
print(rfModel)   # we see that the class.error column values(across d rows) are relatively low when predicting 'No', and the error rate is much higher when predicting 'Yes'.


# Prediction on testing data
pred_rf <- predict(rfModel, testing)

# Confusion matrix
# caret::confusionMatrix(pred_rf, testing$Churn)
table(Predicted = pred_rf, Actual = testing$Churn)

# RF Error rate
plot(rfModel)   # We use this plot to help us determine the number of trees(u can see trees parameter in x-axis).
# As the number of trees increases, the OOB error rate decreases, and then becomes almost constant. We are not able to decrease the OOB error rate after about 100 to 200 trees


# Tune Random Forest Model
t <- tuneRF(training[, -18], training[, 18], stepFactor = 0.5, plot = TRUE,
            ntreeTry = 200, trace = TRUE, improve = 0.05)
t  # We use this plot to give us some ideas on the number of mtry to choose. OOB error rate is at the lowest when mtry is 2. Therefore, we choose mtry = 2.


# create the Random Forest Model after Tuning and getting d optimal mtry value
rfModel_new <- randomForest(Churn ~., data = training, ntree = 200,
                            mtry = 2, importance = TRUE, proximity = TRUE)
print(rfModel_new)


# Prediction on testing data
pred_rf_new <- predict(rfModel_new, testing)

# Confusion matrix
#caret::confusionMatrix(pred_rf_new, testing$Churn)
table(Predicted = pred_rf_new, Actual = testing$Churn)


# Random Forest Feature Importance: displaying d columns/variable in order of importance to d model
varImpPlot(rfModel_new, sort=T, n.var = 10, main = 'Top 10 Feature Importance')











