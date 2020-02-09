library(tidyverse)
library(e1071)
library(reshape2)
library(GGally)
library(caret)
library(xgboost)
library(glmnet)
library(flexclust)

houses_train<-read.csv('houses/houses_train.csv')
houses_test<-read.csv('houses/houses_test.csv')

# Price is skewed to the right, use a log tranformation to make better predictions
# As residuals will be normally distributed, errors in predicting 
# expensive/cheap houses will affect the result equally.
houses_train %>%
  ggplot(aes(log(SalePrice))) +
  geom_histogram()


# create a single dataset to do some data transformations
houses_train$SalePrice<-log(houses_train$SalePrice)
houses_test$SalePrice<-NA
houses_full<-rbind(houses_train, houses_test)

# 2919 rows and 81 variables
str(houses_full)
# Some missing values need to be handled
summary(houses_full)

sort(sapply(houses_full, function(x) { sum(is.na(x)) }))

# some variables are useless (too many repeated)
# mssubclass is categorical, YrSold too because only 5 years
# KitchenAbvGr influences negatively (check a plot or lm) which is not true
houses_full %>% select_if(is.numeric)
houses_full %>% select_if(is.factor) %>% summary
houses_full<-houses_full %>%
  select(-c(Utilities, Street, Fence, Alley, MiscFeature,
            FireplaceQu, RoofMatl, Condition2, Heating, KitchenAbvGr)) %>%
  mutate(MSSubClass = factor(MSSubClass), YrSold = factor(YrSold), MoSold = factor(MoSold))

# mszoning is related to mssubclass, set the most common
houses_full %>% filter(is.na(MSZoning)) %>% select(Id, MSSubClass)
table(houses_full$MSZoning, houses_full$MSSubClass)
houses_full$MSZoning[c(1916, 2251)]<-'RM'
houses_full$MSZoning[c(2217, 2905)]<-'RL'

# neighborhood can be used to estimate LotFrontage
houses_full %>% ggplot(aes(Neighborhood, LotFrontage)) + geom_boxplot() + coord_flip()
houses_full$LotFrontage<-houses_full %>%
  group_by(Neighborhood) %>%
  summarise(m = median(LotFrontage, na.rm = T)) %>%
  right_join(houses_full, by = 'Neighborhood') %>%
  mutate(LotFrontage = ifelse(is.na(LotFrontage), m, LotFrontage)) %>%
  pull(LotFrontage)

# replace remaining NA with the most common value or set it as missing
houses_full<-houses_full %>%
  replace_na(list(Exterior1st = 'VinylSd', Exterior2nd = 'VinylSd', Electrical = 'SBrkr',
                  KitchenQual = 'TA', Functional = 'Typ', SaleType = 'WD')) %>%
  mutate_if(is.factor, fct_explicit_na, na_level = 'None') %>%
  mutate_if(is.numeric, ~ ifelse(is.na(.), 0, .))

# add features
houses_full<-houses_full %>%
  mutate(TotalPorchSF = OpenPorchSF + X3SsnPorch + EnclosedPorch + ScreenPorch + WoodDeckSF,
         Remodded = YearBuilt == YearRemodAdd)

# log transform some right (positive) skewed features
houses_full %>% summarise_if(is.numeric, skewness) %>% sort
houses_full %>% mutate_if(is.numeric, log1p) %>% summarise_if(is.numeric, skewness) %>% sort

hist(houses_full$X1stFlrSF)
hist(log1p(houses_full$X1stFlrSF))

houses_full<-houses_full %>%
  mutate(X1stFlrSF = log1p(X1stFlrSF), GrLivArea = log1p(GrLivArea), LotArea = log1p(LotArea))

# replace ordinal variables
ordinal_quality<-c('None' = 0, 'Po' = 1, 'Fa' = 2, 'TA' = 3, 'Gd' = 4, 'Ex' = 5)
houses_full<-houses_full %>%
  mutate(KitchenQual = recode(KitchenQual, !!!ordinal_quality),
         HeatingQC = recode(HeatingQC, !!!ordinal_quality),
         ExterQual = recode(ExterQual, !!!ordinal_quality),
         GarageQual = recode(GarageQual, !!!ordinal_quality),
         BsmtQual = recode(BsmtQual, !!!ordinal_quality),
         ExterCond = recode(ExterCond, !!!ordinal_quality),
         GarageCond = recode(GarageCond, !!!ordinal_quality),
         BsmtCond = recode(BsmtCond, !!!ordinal_quality),
         PoolQC = recode(PoolQC, !!!ordinal_quality),
         LandSlope = recode(LandSlope, 'Sev'=0, 'Mod'=1, 'Gtl'=2),
         GarageFinish = recode(GarageFinish, 'None' = 0, 'Unf' = 1, 'RFn' = 2, 'Fin' = 3),
         BsmtExposure = recode(BsmtExposure, 'None' = 0, 'No' = 1, 'Mn' = 2, 'Av' = 3, 'Gd' = 4),
         Functional = recode(Functional, 'Sal' = 0, 'Sev' = 1, 'Maj2' = 2, 'Maj1' = 3, 'Mod' = 4,
                             'Min2' = 5, 'Min1' = 6, 'Typ' = 7))



# Split full dataset
nrow(houses_train)
houses_train<-houses_full[1:1460,]
houses_test<-houses_full[1461:2919,]
houses_train$Id<-NULL


# Very high SalePrice correlation with OverallQual and GrLivarea
relevant_numeric<-houses_train %>%
  select_if(is.numeric) %>%
  cor %>%
  round(2) %>%
  melt %>%
  filter(Var1 == 'SalePrice' & abs(value) > 0.5) %>%
  arrange(value) %>%
  pull(Var2) %>%
  as.vector

houses_train %>%
  select(relevant_numeric) %>%
  ggcorr(label = T, hjust = 0.85, layout.exp = 2, label_round = 2)


# Removing these outliers to improve RMSE
houses_train %>%
  ggplot(aes(GrLivArea, SalePrice)) +
  geom_point() +
  geom_smooth() +
  geom_vline(xintercept = 8.45)
houses_train<-houses_train %>% filter(GrLivArea < 8.45 & TotalBsmtSF < 3200)

set.seed(123)
train_control<-trainControl(method = 'repeatedcv', number = 10, repeats = 3)
(linear_model<-train(SalePrice ~ TotalBsmtSF + X1stFlrSF + X2ndFlrSF + TotalPorchSF + GrLivArea  +
                       OverallQual + OverallCond + YearBuilt + LotArea + Fireplaces + HeatingQC +
                       SaleCondition + MSZoning + CentralAir + BsmtFullBath + BsmtExposure +
                       GarageCars + Condition1 + KitchenQual + Functional,
                     data = houses_train, trControl = train_control, method = 'lm'))

# Example to calculate out of sample R2 and RMSE
##baseline -> test RMSE = 0.34, R2 = 0
#(linear_model.pred<-mean(train$SalePrice))
##linear model -> test RMSE = 0.14, R2 = 0.82
#linear_model.pred<-predict(linear_model, test)
#SSE<-sum((test$SalePrice - linear_model.pred)^2)
#SST<-sum((test$SalePrice - mean(train$SalePrice))^2)
#(testRsquared<-1-SSE/SST)
#(testRMSE<-sqrt(SSE/nrow(test)))

# RMSE       Rsquared      
# 0.1147255  0.9176931
getTrainPerf(linear_model)
summary(linear_model)
postResample(predict(linear_model), houses_train$SalePrice)


# lasso regression
set.seed(123)
(lasso_model<-train(SalePrice ~ ., data = houses_train, 
                    trControl = train_control, method = 'glmnet',
                    tuneGrid = expand.grid(alpha = 1, lambda = seq(0.001, 0.1, by = 0.0005))))
# RMSE       Rsquared      
# 0.1118318  0.9219574
getTrainPerf(lasso_model)
varImp(lasso_model)


# ridge regression
set.seed(123)
(ridge_model<-train(SalePrice ~ ., data = houses_train, trControl = train_control, method = 'glmnet',
                    tuneGrid = expand.grid(alpha = 0, lambda = seq(0.01, 1, by = 0.005))))
# RMSE       Rsquared      
# 0.1147054  0.9176909
getTrainPerf(ridge_model)
varImp(ridge_model)


# elasticnet
set.seed(123)
(elastic_model<-train(SalePrice ~ ., data = houses_train, trControl = train_control, method = 'glmnet',
                      tuneGrid = expand.grid(alpha = seq(0, 1, 0.05), lambda = seq(0.01, 0.2, 0.005))))
# RMSE       Rsquared      
# 0.1114855  0.9225483
getTrainPerf(elastic_model)
varImp(elastic_model)


# xgb - max_depth = 4, gamma = 0, min_child_weight = 5
set.seed(123)
xgb_grid <- expand.grid(
  nrounds = c(1000),
  max_depth = c(3, 4),
  eta = c(0.01),
  gamma = c(0, 1),
  colsample_bytree = c(0.7),
  subsample = c(0.7),
  min_child_weight = c(1, 5)
)
(xgb_model<-train(SalePrice ~ ., data = houses_train,
                  method = 'xgbTree', tuneGrid = xgb_grid, trControl = train_control))
# RMSE       Rsquared      
# 0.1177898  0.9135709
getTrainPerf(xgb_model)
varImp(xgb_model)


# Export data and get performance
# 0.12897 linear
solution<-data.frame(ID = houses_test$Id, SalePrice = exp(predict(linear_model, houses_test)))
write.csv(solution, file = 'houses-linear.csv', row.names = F)

# 0.11911 - lasso
solution<-data.frame(ID = houses_test$Id, SalePrice = exp(predict(lasso_model, houses_test)))
write.csv(solution, file = 'houses-lasso.csv', row.names = F)

# 0.12142 - ridge
solution<-data.frame(ID = houses_test$Id, SalePrice = exp(predict(ridge_model, houses_test)))
write.csv(solution, file = 'houses-ridge.csv', row.names = F)

# 0.11977 - elasticnet
solution<-data.frame(ID = houses_test$Id, SalePrice = exp(predict(elastic_model, houses_test)))
write.csv(solution, file = 'houses-elastic.csv', row.names = F)

# 0.12640 - xgb
solution<-data.frame(ID = houses_test$Id, SalePrice = exp(predict(xgb_model, houses_test)))
write.csv(solution, file = 'houses-xgb.csv', row.names = F)



# Try to improve model by clustering houses and applying a model for each cluster
# Need to scale features with algorithms that compute distances
# Have to use numeric variables and can't use predicted variable
preproc<-preProcess(houses_train %>% select(relevant_numeric) %>% select(-SalePrice))
normTrain<-predict(preproc, houses_train %>% select(relevant_numeric) %>% select(-SalePrice))
normTest<-predict(preproc, houses_test %>% select(relevant_numeric) %>% select(-SalePrice))
# Hierarchical clustering - should use 2, 3 or 4 clusters
distances<-dist(normTrain, method = 'euclidean')
hc<-hclust(distances, method = 'ward.D')
plot(hc)
rect.hclust(hc, k = 3, border = 'red')
clusterGroups<-cutree(hc, k = 2)
table(clusterGroups)
houseClusters<-split(houses_train, clusterGroups)

# Scree plot using the elbow method to determine ideal number of clusters - 2, 3 or 4
set.seed(123)
nclusters<-1:10
sumWithinss<-sapply(nclusters, function(x) { 
  sum(kmeans(normTrain, centers = x, iter.max = 1000)$withinss) 
})
plot(nclusters, sumWithinss, type='b')

set.seed(123)
km<-kmeans(normTrain, centers = 2, iter.max = 1000)
table(km$cluster)
kmClusters<-split(houses_train, km$cluster)

# Compare clusters
table(clusterGroups, km$cluster)

# PCA and visualization - similar to factoextra package
pc<-prcomp(normTrain, scale = FALSE, center = FALSE)
PoV<-pc$sdev^2/sum(pc$sdev^2)
head(PoV)

pca_rep <- data.frame(pc1 = pc$x[,1],
                      pc2 = pc$x[,2],
                      clust_id = as.factor(km$cluster))

ggplot(pca_rep, aes(pc1, pc2, color = clust_id, shape = clust_id)) +
  geom_point() +
  stat_ellipse(geom = 'polygon', mapping = aes(fill = clust_id), alpha = 0.1) +
  scale_shape_manual(values=seq(0,2))

# predict cluster
km.kcca = as.kcca(km, normTrain)
clusterTrain<-predict(km.kcca)
clusterTest<-predict(km.kcca, newdata=normTest)
prop.table(table(clusterTrain))
prop.table(table(clusterTest))

train1<-subset(houses_train, clusterTrain == 1)
train2<-subset(houses_train, clusterTrain == 2)
test1<-subset(houses_test, clusterTest == 1)
test2<-subset(houses_test, clusterTest == 2)

set.seed(123)
model1<-train(SalePrice ~ ., data = train1, trControl = train_control, method = 'glmnet',
              tuneGrid = expand.grid(alpha = 1, lambda = seq(0.001, 0.1, by = 0.0005)))
model2<-train(SalePrice ~ ., data = train2, trControl = train_control, method = 'glmnet',
              tuneGrid = expand.grid(alpha = 1, lambda = seq(0.001, 0.1, by = 0.0005)))
getTrainPerf(model1)
getTrainPerf(model2)

# Example of estimating accuracy for regression/classification when we have a test set with values:
# postResample(c(model1.pred, model2.pred), c(test1$SalePrice, test2$SalePrice))
# AllPredictions = c(predictLog1, predictLog2, predictLog3)
# AllOutcomes = c(test1$Outcome, test2$Outcome, test3$Outcome)
# table(AllOutcomes, AllPredictions > 0.5)

model1.pred<-data.frame(SalePrice = exp(predict(model1, test1))) %>% rownames_to_column('ID')
model2.pred<-data.frame(SalePrice = exp(predict(model2, test2))) %>% rownames_to_column('ID')

# 0.11897 - a small improvement
solution<-rbind(model1.pred, model2.pred) %>% arrange(ID) %>% mutate(ID = as.numeric(ID))
write.csv(solution, file = 'houses-clustered.csv', row.names = F)


# 0.11765 (top 20%) - combining all models gives a small improvement, more improvements could be made
solution<-data.frame(ID = houses_test$Id,
                     SalePrice = exp(predict(lasso_model, houses_test)) * 0.25 +
                       solution$SalePrice * 0.25 +
                       exp(predict(ridge_model, houses_test)) * 0.15 +
                       exp(predict(elastic_model, houses_test)) * 0.15 +
                       exp(predict(xgb_model, houses_test)) * 0.1 +
                       exp(predict(linear_model, houses_test)) * 0.1)
write.csv(solution, file = 'houses-combined.csv', row.names = F)
