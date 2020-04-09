library(tidyverse)
library(stringi)
library(quanteda)
library(caret)

twitter_train<-as_tibble(read.csv('twitter/train.csv', stringsAsFactors = F, encoding = 'UTF-8'))
twitter_test<-as_tibble(read.csv('twitter/test.csv', stringsAsFactors = F, encoding = 'UTF-8'))

# 110 tweets are not unique
nrow(twitter_train)
length(unique(twitter_train$text))
prop.table(table(twitter_train$target))

# keyword may not be present in text
glimpse(twitter_train)
table(twitter_train$keyword, twitter_train$target)

# use target mode to deal with duplicate texts
mode <- function(x){
  as.numeric(names(which.max(table(x))))
}
twitter_train<-twitter_train %>%
  group_by(text) %>% 
  mutate(target = mode(target)) %>%
  ungroup() %>%
  distinct(text, .keep_all = TRUE)

# add new features
add_features<-function(twitter_data){
  twitter_data %>%
    mutate(Hash = str_count(text, '#.+'), Ref = str_count(text, '@\\w+'),
           Url = str_count(text, 'https?://'), Words = str_count(text, '\\S+'),
           Punct = str_count(text, '[[:punct:]]+'), Caps = str_count(text, '\\b[A-Z]{2,}\\b')) %>%
    select(Hash, Ref, Url, Punct, Caps)
}

# vectorize text
vectorize_text<-function(twitter_data) {
  twitter_data %>%
    # unite('text', keyword, text, sep = ' ') %>%
    pull(text) %>%
    str_replace_all('%20', ' ') %>%
    stri_trans_general('Latin-ASCII') %>%
    tokens(remove_punct = T, remove_symbols = T, remove_url = T, remove_numbers = T) %>%
    tokens_remove(c('amp', 'u_', 'rt', 'uªs', 'bombed', 'ablaze', 'emergency', stopwords('en'))) %>%
    tokens_select(min_nchar = 2) %>%
    tokens_tolower() %>%
    # tokens_wordstem(language = 'en') %>%
    # tokens_ngrams(n = 1:2) %>%
    dfm()
}

dfm_train<-vectorize_text(twitter_train) %>% dfm_trim(min_docfreq = 15)
dim(dfm_train)
dfm_test<-vectorize_text(twitter_test)
dim(dfm_test)
dfm_matched<-dfm_match(dfm_test, features = featnames(dfm_train))
dim(dfm_matched)

# convert bag of words to data frame and add more variables
# could use tfidf weighting but worst performance for this dataset - dfm_tfidf(dfm, scheme_tf = 'prop')
df_train<-convert(dfm_train, 'data.frame')
df_test<-convert(dfm_matched, 'data.frame')
df_train<-df_train %>%
  mutate(Target = as.factor(twitter_train$target)) %>%
  cbind(add_features(twitter_train))
df_test<-df_test %>% cbind(add_features(twitter_test))

# wordcloud
set.seed(123)
dfm_train$Target<-df_train$Target
levels(dfm_train$Target)<-c('Not disaster', 'Disaster')
dfm_train %>%
  dfm(groups = 'Target') %>%
  textplot_wordcloud(rotation = F, max_words = 200, random_order = F,
                     comparison = T, color = c('blue', 'red'))


# words near the line are used with about equal frequency - https://www.tidytextmining.com/twitter.html
# add some of them to the stopwords list
library(scales)
dfm_train %>%
  dfm_group(groups = 'Target') %>%
  dfm_weight(scheme = 'prop') %>%
  textstat_frequency(groups = 'Target') %>%
  select(feature, frequency, group) %>%
  mutate(group = as.factor(group)) %>%
  spread(group, frequency, fill = 0) %>%
  ggplot(aes(Disaster, `Not disaster`)) +
    geom_jitter(alpha = 0.1, size = 2.5, width = 0.25, height = 0.25) +
    geom_text(aes(label = feature), check_overlap = T, vjust = 1.5) +
    scale_x_log10(labels = percent_format()) +
    scale_y_log10(labels = percent_format()) +
    geom_abline(color = 'red')

# use F1 score instead of accuracy to evaluate
control<-trainControl(method = 'cv', number = 5, summaryFunction = prSummary)

# Linear SVM - 0.8324335 cv F1 (0.80368 on test) - stemming/ngrams and append keywords no improvement
# cost = 0.15, Loss = L1 and weight = 1
set.seed(123)
# preProc<-preProcess(df_train, method = c('center', 'scale')) # required if using different scales
grid<-expand.grid(cost = seq(0.05, 0.2, 0.05), Loss = c('L1', 'L2'), weight = seq(0.5, 2, 0.5))
(svm_train<-train(Target ~ ., data = df_train, method = 'svmLinearWeights2',
                  metric = 'F', tuneGrid = grid, trControl = control))
getTrainPerf(svm_train)
ggplot(svm_train)

solution<-data.frame(id = twitter_test$id, target = predict(svm_train, df_test))
write.csv(solution, file = 'solution.csv', row.names = F)

# keyword probabilities can improve predictions
solution$keyword = twitter_test$keyword

keywords_summary<-twitter_train %>%
  count(target, keyword) %>%
  spread(target, n) %>%
  replace_na(list(`1` = 0, `0` = 0)) %>%
  mutate(total = `1` + `0`, p = `1` / total)

View(keywords_summary)

bad_keywords<-keywords_summary %>% filter(p > 0.9) %>% pull(keyword)

good_keywords<-keywords_summary %>% filter(p < 0.1) %>% pull(keyword)

solution2<-solution %>%
  mutate(new_target = ifelse(keyword %in% bad_keywords, '1',
                             ifelse(keyword %in% good_keywords, '0', as.character(target))))

solution2 %>%
  filter(new_target != target & keyword %in% c(bad_keywords, good_keywords)) %>%
  View()

# Final test score: 0.80674 top 38%
solution2 %>%
  select(id, new_target) %>%
  rename(target = new_target) %>%
  write.csv(file = 'solution2.csv', row.names = F)

# ensemble model: https://www.kaggle.com/barun2104/nlp-with-disaster-eda-tf-idf-svd-ensemble
