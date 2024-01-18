install.packages("c50")
install.packages("gmodels")
install.packages("caret")
install.packages("naivebayes")
install.packages("performanceEstimation")
library(C50)
library(gmodels)
library(caret)
library(naivebayes)
library(performanceEstimation)

#----------------------Data Preparation-----------------------------------------
rt1_dataset <- read.csv('cleaned.csv', stringsAsFactors = TRUE)
rt1_dataset$Accident_severity[rt1_dataset$Accident_severity == 0] <- "Fatal Injury"
rt1_dataset$Accident_severity[rt1_dataset$Accident_severity == 1] <- "Serious Injury"
rt1_dataset$Accident_severity[rt1_dataset$Accident_severity == 2] <- "Slight Injury"
rt1_dataset$Accident_severity <- as.factor(rt1_dataset$Accident_severity)
rt1_dataset <- subset(rt1_dataset, select = -Pedestrian_movement)

set.seed(123)
data_split <- sample(nrow(rt1_dataset), nrow(rt1_dataset) * 0.8)
train_data <- rt1_dataset[data_split, ]
test_data <- rt1_dataset[-data_split, ]
table(train_data$Accident_severity)
prop.table(table(train_data$Accident_severity))

# Smote
train_smote <- smote(Accident_severity ~ ., data = train_data, perc.over = 5, perc.under = 4)
table(train_smote$Accident_severity)
prop.table(table(train_smote$Accident_severity))

#----------------------Decision Tree--------------------------------------------
rta1Tree <- C5.0(Accident_severity ~ ., data = train_data)
rta4Tree <- C5.0(Accident_severity ~ ., data = train_smote)
pred1 <- predict(rta1Tree, test_data)
pred4 <- predict(rta4Tree, test_data)
confusionMatrix(pred1, test_data$Accident_severity, mode = "prec_recall")
confusionMatrix(pred4, test_data$Accident_severity, mode = "prec_recall")

#-----------------------Naive Bayes---------------------------------------------
(model1 <- naive_bayes(Accident_severity ~ ., data = train_data, laplace = 1))
(model4 <- naive_bayes(Accident_severity ~ ., data = train_smote, laplace = 1))
p1 <- predict(model1, test_data[, -ncol(test_data)])
p4 <- predict(model4, test_data[, -ncol(test_data)])
confusionMatrix(p1, test_data$Accident_severity, mode = "prec_recall")
confusionMatrix(p4, test_data$Accident_severity, mode = "prec_recall")

