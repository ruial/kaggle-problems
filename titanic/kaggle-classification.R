library(tidyverse)
library(GGally)
library(mice)
library(caTools)
library(ROCR)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(xgboost)

titanic_train<-read.csv('titanic/titanic_train.csv')
titanic_test<-read.csv('titanic/titanic_test.csv')

# create a single dataset to do some data transformations
titanic_test$Survived<-NA
titanic_full<-rbind(titanic_train, titanic_test)

# 1309 rows and 12 variables
str(titanic_full)
# Age contains missing values, 1 Fare is NA
summary(titanic_full)
# View data
View(titanic_full)

# Transform some continuous variables to categorical
titanic_full<-titanic_full %>%
  mutate(Survived = factor(Survived), Pclass = factor(Pclass))

# Impute fare and embarked values using medians (less sensitive to outliers)
titanic_full %>% filter(is.na(Fare) | Embarked == '')
table(titanic_full$Pclass, titanic_full$Embarked)
table(titanic_full$Cabin != '', titanic_full$Pclass)

titanic_full %>%
    group_by(Pclass, Embarked) %>%
    summarise(median = median(Fare, na.rm = T), mean = mean(Fare, na.rm = T), n = n())

titanic_full<-titanic_full %>%
  replace_na(list(Fare = 8.05)) %>%
  mutate(Embarked = fct_collapse(Embarked, 'C' = ''))


# Feature engineering
titanic_full$Title<-gsub('\\..*', '', gsub('.*, ', '', titanic_full$Name))
table(titanic_full$Title) %>% sort() %>% rev()

titanic_full<-titanic_full %>%
  mutate(Title = factor(Title)) %>%
  mutate(Title = fct_collapse(Title, 
                              'Miss' = c('Mlle', 'Ms'), 
                              'Mrs' = 'Mme', 
                              'Other' = c( 'Major', 'Dr', 'Capt', 'Col', 'Rev',
                                           'Lady', 'Dona', 'the Countess', 'Don', 'Sir', 'Jonkheer')))

titanic_full<-titanic_full %>%
  mutate(Fsize = SibSp + Parch + 1) %>%
  add_count(Ticket) %>%
  rename(Nticket = n) %>%
  mutate(Fsize = ifelse(Fsize > Nticket, Fsize, Nticket)) %>%
  select(-Nticket)


# Use mice package to handle missing values at random after having the Title feature
set.seed(123)
imputed<-complete(mice(titanic_full[, !names(titanic_full) %in% c('PassengerId', 'Survived', 'Name',
                                                                  'Ticket', 'Cabin')],
                       method = 'rf'))
titanic_full$Age<-imputed$Age


# Split full dataset
nrow(titanic_train)
titanic_train<-titanic_full[1:891,]
titanic_test<-titanic_full[892:1309,]

# Remove some features
titanic_train$PassengerId<-NULL
titanic_train$Name<-NULL
titanic_train$Ticket<-NULL
titanic_train$Cabin<-NULL
titanic_train$SibSp<-NULL
titanic_train$Parch<-NULL

# Basic EDA
titanic_train %>% ggpairs(title = 'EDA', mapping = aes(color = Survived))

data.matrix(titanic_train) %>% ggcorr(label = T)

titanic_train %>%
  ggplot(aes(Title, Age, color = Survived, shape = Sex, size = Fsize)) +
  geom_jitter(width = 0.3, height = 0, alpha = 0.7) +
  facet_grid(Embarked~Pclass)

titanic_train %>%
  ggplot(aes(Fare + 1, fill = Survived)) +
  geom_histogram() +
  scale_x_log10() +
  facet_grid(~Pclass)

# 62% of people died died
prop.table(table(titanic_train$Survived))
# Baseline model - 79% accuracy if predicting women survived and men didn't
prop.table(table(titanic_train$Survived, titanic_train$Sex))


# Logistic regression model
# Split the training data into a train and test set to estimate out of sample accuracy
# Seed necessary for reproducibility in splits, cross validation and some models 
set.seed(123)
split<-sample.split(titanic_train$Survived, SplitRatio = 0.7)
train<-subset(titanic_train, split == TRUE)
test<-subset(titanic_train, split == FALSE)

#log_model<-glm(Survived ~ ., data = train, family = 'binomial')
#log_model$xlevels$Title <- union(log_model$xlevels$Title, titanic_full$Title)
log_model<-glm(Survived ~ log(Fare + 1) + Title + Fsize + Pclass + Age + Embarked,
               data = train, family = 'binomial')
summary(log_model)

# 84% training accuracy
log_model.pred<-predict(log_model, type='response')
table(train$Survived, log_model.pred > 0.5)
(340+185)/nrow(train)

# 82% test accuracy
log_model.pred<-predict(log_model, newdata = test, type='response')
table(test$Survived, log_model.pred > 0.5)
(146+74)/nrow(test)

# ROC curve
ROCRpred<-prediction(predict(log_model), train$Survived)
ROCRperf<-performance(ROCRpred, 'tpr', 'fpr')
plot(ROCRperf)
plot(ROCRperf, colorize=TRUE)

# 0.86 AUC
ROCRpredTest<-prediction(log_model.pred, test$Survived)
as.numeric(performance(ROCRpredTest, 'auc')@y.values)

# Logistic model using full training data
set.seed(123)
log_train<-train(Survived ~ log1p(Fare) + Title + Fsize + Pclass + Age + Embarked, data = titanic_train,
                 method = 'glm', trControl = trainControl(method = 'repeatedcv', number = 10, repeats = 3))
summary(log_train)
# 84% training accuracy, 83% cross validation accuracy
getTrainPerf(log_train)
# can't do predictions when glm train formula has transformations, create new model or mutate data before
# confusionMatrix(predict(log_train), titanic_train$Survived)
log_model<-glm(Survived ~ log1p(Fare) + Title + Fsize + Pclass + Age + Embarked,
               data = titanic_train, family = 'binomial')
confusionMatrix(factor(ifelse(predict(log_model, type = 'response') > 0.5, 1, 0)), titanic_train$Survived)


# Classification and regression tree model 
# Remove some correlated variables to improve the model (prevent overfitting)
# Use cross validation to estimate out of sample accuracy instead of spliting the dataset
tree_model<-rpart(Survived ~ ., data = subset(titanic_train, select = c(-Fare, -Age)), 
                  method = 'class', cp = 0.001)
prp(tree_model)

# cross validation to obtain cp hyperparameter
set.seed(123)
cartGrid<-expand.grid(.cp = seq(0.001,0.1,0.01))
numFolds<-trainControl(method = 'cv', number = 10)
# rpart/randomforest/naivebayes unlike other functions in R, do not convert factors to dummy variables
# the tree isn't the same as rpart when using the train formula, to prevent this xy can be used instead
(tree_train<-train(Survived ~ . -Fare -Age, 
                   data = titanic_train, method = 'rpart', trControl = numFolds, tuneGrid = cartGrid))
ggplot(tree_train)
getTrainPerf(tree_train)
prp(tree_train$finalModel, varlen = 0)
rpart.plot(prune(tree_train$finalModel, cp = 0.005))

# 84% training accuracy, 83% cross validation accuracy
confusionMatrix(predict(tree_train, type = 'raw'), titanic_train$Survived)


# Random forest model
set.seed(123)
rf_model<-randomForest(Survived ~ ., data = titanic_train)
confusionMatrix(predict(rf_model, type = 'class'), titanic_train$Survived)

# hyperparameter tuning using repeated k-fold cross validation
set.seed(123)
control<-trainControl(method='repeatedcv', number = 10, repeats = 3)
(rf_train<-train(Survived ~ . -Fare -Age,
                 data = titanic_train, method = 'rf', tuneLength = 5, trControl = control))

varImpPlot(rf_train$finalModel)
# 83% training accuracy, 83% cross validation accuracy with mtry = 2
confusionMatrix(predict(rf_train), titanic_train$Survived)


# XGBoost model
set.seed(123)
xgbGrid <- expand.grid(
  nrounds = c(500),                # 100 default
  max_depth = c(2, 3, 4, 5),       # 6 default
  eta = c(0.01, 0.1, 0.15),        # 0.3 default
  gamma = c(0, 1),                 # 0 default
  colsample_bytree = c(0.8),       # 1 default
  subsample = c(0.8),              # 0.5 default
  min_child_weight = c(1, 3)       # 1 default
)
(xgb_train<-train(Survived ~ . -Fare -Age,
                 data = titanic_train, method = 'xgbTree', tuneGrid = xgbGrid, trControl = numFolds))
# 84% training accuracy, 83% cross validation accuracy
confusionMatrix(predict(xgb_train), titanic_train$Survived)
summary(xgb_train$results$Accuracy)


# Logistic regression - 79% final accuracy
solution<-data.frame(PassengerID = titanic_test$PassengerId, Survived = 
                       ifelse(predict(log_model, titanic_test, type = 'response') > 0.5, 1, 0))
write.csv(solution, file = 'solution-log.csv', row.names = F)

# Classification tree - 80% final accuracy -> 0.80382 (top 10%)
solution<-data.frame(PassengerID = titanic_test$PassengerId, Survived = predict(tree_train, titanic_test))
write.csv(solution, file = 'solution-tree.csv', row.names = F)

# Combine 2 models with weighted average probabilities at 50%
solution<-data.frame(RealSurvived = titanic_train$Survived,
                              SurvivedLogistic = predict(log_model, titanic_train, type = 'response'),
                              SurvivedTree = predict(tree_train, type = 'prob')[,2]) %>%
  mutate(SurvivedProb = SurvivedLogistic * 0.5 + SurvivedTree * 0.5,
         Survived = ifelse(SurvivedProb > 0.5, 1, 0))

confusionMatrix(factor(solution$Survived), titanic_train$Survived) # 84% accuracy

solution<-data.frame(PassengerID = titanic_test$PassengerId,
                     Name = titanic_test$Name,
                     SurvivedLogistic = predict(log_model, titanic_test, type = 'response'),
                     SurvivedTree = predict(tree_train, titanic_test, type = 'prob')[,2]) %>%
  mutate(SurvivedProb = SurvivedLogistic * 0.5 + SurvivedTree * 0.5,
         Survived = ifelse(SurvivedProb > 0.5, 1, 0))

solution %>% filter((round(SurvivedLogistic) != round(SurvivedTree)))

# 80% accuracy - no improvement over CART
write.csv(solution %>% select(PassengerID, Survived), file = 'solution-combined.csv', row.names = F)

# The CART model has the best accuracy and is very interpretable
# Logistic regression is a simple model that provides similar accuracy
# XGB/RF are more complex and don't improve results (78-79%)
