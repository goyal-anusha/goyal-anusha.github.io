---
title: "Final Final Project"
format: html
editor: visual
---

Group 13: Asavari Ahuja, Yoon Choi, Anusha Goyal, Dave John

APAN 5205: Applied Analytics Frameworks and Methods 2

# **Reviewing Success: IMDb's Predictive Power**

## **Research problem**

Box office success is a multifaceted concept that embodies both financial returns and cultural impact. It is essential to recognize that the criteria for box office success vary widely across films and industry; box office rankings, return on investment, critical reception, audience attendance, and international recognition are among the many indicators that determine box office success. Each film has its own measure of success at the box office. However, for the purpose of this research and to effectively compare box office success, we measure the box office success as a film's user ratings on IMDb. The IMDb user rating scale goes from one to ten, and a higher score is a more successful film. By conducting a predictive analysis using the reviews' sentiment analysis, we aim to predict the user ratings of the movies.

\
Further, we seek to compare the predicted user ratings for directors of different demographics, to determine whether introducing the director demographic variable would more accurately predict the user generated movie scores. For the purpose of the research, we hypothesize that conventional demographic characteristics for movie directors, such as being "white" and "male" are positively correlated with a movie's box office success. Through this study, we seek to contribute to a more nuanced understanding of the dynamics of the contemporary film industry. Prior to conducting any exhaustive research on the topic at hand, we initially identified the following words from the movie reviews to be positively correlated with a movie's box office success: realistic, clever, thrilling, exciting, witty, legendary, and relatable.

## Literature Review

Since the beginning of Cinema, ratings, reviews and critiques have been a fundamental aspect of the film industry. These opinion statements served as not only a review of the movie's quality, but also a way of demonstrating one's expertise, thereby proving one's credentials of thriving in the industry. In earlier years of movie critique, movie criticism mainly included technical aspects and narrative effectiveness. However, over the years, movie critique evolved from just providing a scholarly analysis of filmmaking aspects and artistic merits. In recent years, movie criticism has expanded beyond industry experts to include the perspectives of everyday individuals, largely due to technological advancements and the availability of rating platforms such as IMDb. Due to the technological advancements that enabled online movie reviews and ratings, the practice of film criticism has become more dynamic. 

Traditionally, movie critics were seen as influencers or predictors of the movie's success or failure at the box office. Their reviews not only influenced pre-sales but also played a pivotal role in shaping the overall credibility of a movie within the industry (Boatwright et al., 2007). IMDb, Metacritic and Rotten Tomatoes are the main mediums to read reviews to make a judgment on a given movie. Access to these platforms have made writing and reading these reviews very convenient for avid movie watchers. While Metacritic and Rotten Tomatoes rely on certified critics to aggregate a review and rating of a movie, IMDb also relies on its registered users to provide a review on a movie. Therefore, there's always a diverse set of opinions --- either positive or negative, professional or informal --- providing a nuanced perspective on movies.

\
Online movie reviews have shown great influence on the box office numbers. This impact was observed during the pandemic in 2020, when the United States and Canada recorded a decline in box office revenue to \$2.1 billion dollars. However, as the world gradually restored to normalcy, there was a substantial increase in box office revenue in the United States and Canada in 2023 to \$8.91 billion dollars (Statista Research Department, 2024). A research paper published in 2023 investigated whether movie critics' reviews of pre-sales movies can accurately predict the film's box office success. The study revealed that the movie critics' "positive (negative) pre-release movie reviews provide a strong predictive signal that the movie will turn out to be a flop (success)"(Loupos et al., 2023). Despite having expertise in the field of movie critiquing, professional reviewers were not able to provide a good prediction of a movie's box office success. Therefore, the researchers went on to further explore the writing styles of non-harbinger and harbinger critics, with the latter often reflecting personal perspectives rather than serving as a voice for the audience (Loupos et al., 2023). Furthermore, an empirical investigation was conducted to assess the impact of movie reviews at each stage of a movie's lifecycle. The findings indicated that critics' reviews held the most impact during the movie's initial release, and then were gradually overpowered by the impact of word-of-mouth (Basuroy et al., 2006). Therefore, through this analysis, we wanted to further explore the impact online reviews from platforms such as IMDb have on the overall audience ratings of the movie.

## **Research Questions**

Our research seeks to explore the following questions:

1.  Can IMBD online reviews serve as a reliable indicator of a film's box office success?

2.  How do director demographics influence movie's success, and does it make the overall prediction model more accurate?

## **Data Source Choices**

1.  **Official IMDb non-commercial datasets**\
    These datasets were taken from the official IMDb website, which consisted of a vast number of tables with attributes about various movies. 

    1.  **title.akas.tsv.gz:** Contains alternative titles for each media, detailing the title ID, localized title, region, language, and attributes such as whether it's an original title.

    2.  **title.basics.tsv.gz:** Lists basic information about each title, including type, primary and original title, adult classification, release years, runtime, and genres.

    3.  **title.crew.tsv.gz:** Provides information on the directors and writers associated with each title.

    4.  **title.ratings.tsv.gz:** Features the average rating and total number of votes each title has received.

2.  **Reviews dataset\
    **The reviews dataset was used to predict ratings. This is aggregated from a web scraper that uses an unofficial IMDb API. We built this web scraper in python to pull the reviews from the first page of IMDB for any given titleId. After this, it outputs all the information into a separate table. This is the data we used to conduct a sentiment analysis that predicts ratings for different movies. We also used the director and genre columns to identify whether they provide additional information to explain disparities in our model.

### **Data Cleaning and Processing**

In an effort to construct our dataset for movie analysis, we utilized four key TSV files: 'title.akas', 'title.basics', 'title.ratings', and 'title.crew'. These files collectively cover the information we require on movie titles, including alternative names, basic attributes, viewer ratings, and crew details. The initial phase focused on **filtering the datasets** to narrow down the scope to English-language movies. The filtering was necessary due to the sheer size of the TSV files. It would be extremely computationally heavy if we had worked on the entire dataset. We also chose to stick with English since it's the focus of our analysis and that these movies would predominantly have English reviews, thus making them more pertinent for sentiment analysis.

The core of the data processing involved a series of **merging steps**. The 'title.akas' dataset, filtered for English titles, was merged with 'title.basics' to align alternative titles with their fundamental characteristics. This merged dataset was further expanded by incorporating ratings from 'title.ratings' and crew information from 'title.crew', thereby creating a comprehensive dataset that includes a wide array of movie-related information such as titles, regions, languages, start years, runtimes, genres, ratings, votes, and director names.

The next transformative step we took was the **splitting** of the directors and genres fields, which contained multiple values, into distinct columns. This was achieved by expanding each value into separate columns, using 'str.split' thus facilitating a more granular analysis. The final dataset, meticulously structured and cleaned, was exported as a CSV file, ready for the next phase of API scraping.

During our analysis we also had to encounter the issue of categorical variables not being compatible with many models. Therefore, when required these variables were transformed into factors in order to not act as a barricade. Similarly, numerical variables that had been imported as factors were reverted back to numerical, i.e the runtime in minutes. 

This data processing endeavor not only helped to streamline the dataset for English-language movies but also enhanced its analytical utility by disaggregating complex fields into analyzable columns, thereby laying a solid foundation for subsequent in-depth movie data analysis.

### Processing the movie information dataset

The following code illustrates how we cleaned the IMDB official datasets with all the information about the movies.

```{r}
library(readr)
library(tidyverse)
basic = read_tsv("/Users/anusha/Downloads/title.akas.tsv")
titles = read_tsv("/Users/anusha/Downloads/title.basics.tsv")
ratings = read_tsv("/Users/anusha/Downloads/title.ratings.tsv")
crew = read_tsv("/Users/anusha/Downloads/title.crew.tsv")

#Filtering for English movies ( so we can hope to use only English reviews and movies)
basic_filtered = basic[basic$language == "en", ]
titles_filtered = titles[titles$titleType == "movie", ]

#Joining basics and titles
inner_join = merge(basic_filtered, titles_filtered, by.x = "titleId", by.y = "tconst")

#joining the director information and the ratings for the movies
ratings_and_movies = merge(inner_join, ratings, by.x = "titleId", by.y = "tconst")
all_data = merge(ratings_and_movies, crew, by.x = "titleId", by.y = "tconst")

#Keeping only the columns we will need
movie_data_to_keep = c("titleId", "title", "region", "language", "startYear", "runtimeMinutes", "genres", "averageRating", "numVotes", "directors")
movie_data = all_data[, movie_data_to_keep]
```

```{r}
#seperating the directors and genres so we have them in different columns
split_directors <- strsplit(movie_data$directors, ",")
max_names <- max(sapply(split_directors, length))
new_columns <- matrix(NA, nrow = nrow(movie_data), ncol = max_names)

for (i in 1:max_names) {
  new_columns[, i] <- sapply(split_directors, function(x) x[i])
}

colnames(new_columns) <- paste("director", 1:max_names, sep = "_")
movie_data <- cbind(movie_data, new_columns)


#seperating the directors and genres so we have them in different columns
split_genres <- strsplit(movie_data$genres, ",")
max_names_genre <- max(sapply(split_genres, length))
new_columns_genre <- matrix(NA, nrow = nrow(movie_data), ncol = max_names_genre)

for (i in 1:max_names_genre) {
  new_columns_genre[, i] <- sapply(split_genres, function(x) x[i])
}

colnames(new_columns_genre) <- paste("genre", 1:max_names_genre, sep = "_")
movie_data <- cbind(movie_data, new_columns_genre)
```

```{r}
#Converting the dataset from wide to tall format
# Melt the director columns
df_long_directors <- movie_data %>%
  pivot_longer(
    cols = starts_with("director_"),
    names_to = "director_number",
    values_to = "director_id",
    names_prefix = "director_",
    values_drop_na = TRUE
  )

# Melt the genre columns
df_long_genres <- movie_data %>%
  pivot_longer(
    cols = starts_with("genre_"),
    names_to = "genre_number",
    values_to = "genre_name",
    names_prefix = "genre_",
    values_drop_na = TRUE
  )

# You can join these two tall datasets back together by the common identifier (e.g., titleId) if needed
df_long <- full_join(df_long_directors, df_long_genres, by = "titleId", relationship = "many-to-many")


column_names <- colnames(df_long)

# Pairing each column name with its index number
column_indices <- seq_along(column_names)

# Creating a named vector where names are column names and values are indices
named_indices <- setNames(column_indices, column_names)
named_indices

# Keeping only the rows I want
df_long_cleaning <- df_long[ , c(1:10,14:15,91:92)]

str(df_long_cleaning)

# Replacing "\N" with null vaules
df_long_cleaning <- df_long_cleaning %>%
  mutate(across(where(is.character), ~na_if(., "\\N")))

# Check is there are there are any /N's left (sanity check)
counts <- df_long_cleaning %>%
  summarise(across(everything(), ~sum(. == "\\N", na.rm = TRUE)))

# Count Null values in each column (sanity check)
na_counts <- colSums(is.na(df_long_cleaning))
na_counts

# Replace all occurrences of ".x" in the column names with an empty string
colnames(df_long_cleaning) <- gsub("\\.x$", "", colnames(df_long_cleaning))

# Remove the original genres and directors column
df_long_cleaning <- df_long_cleaning[ , c(1:6,8:9,11:14)]

```

### Formatting the data for Reviews from the Python Parser

```{r}
#seperate column 2 to ensure each variable is in its own column
library(tidyr)
library(dplyr)
data = read.csv("/Users/anusha/Desktop/Columbia School/frameworks2/Project/reviews_output.csv")
head(data)
data_separated <- data %>%
  extract(review, into = c("data_review_id", "full_review", "rating_value", "review_date", "reviewer_name", "reviewer_url", "short_review"), 
          regex = "'data-review-id': '(.*?)', 'full_review': '(.*?)', 'rating_value': '(.*?)', 'review_date': '(.*?)', 'reviewer_name': '(.*?)', 'reviewer_url': '(.*?)', 'short_review': '(.*?)'") %>%
  mutate(rating_value = as.numeric(rating_value))

head(data_separated)
```

```{r}
#Remove completely empty entries (which are entries without review_id)
df = subset(data_separated, !is.na(data_review_id))
nrow(df)
df$titleId = df$id
df$id <- seq_along(df[,1])
```

## Initial Exploratory Analysis of the IMDB reviews

```{r}
#Keep only reviews with all the text
long_review = subset(df, full_review != "")
summary(nchar(long_review$full_review))


#Check correlations between review structure and rating
library(stringr)
long_review %>%
  select(full_review)%>%
  mutate(characters = nchar(full_review),
         words = str_count(full_review,pattern='\\S+'),
         sentences = str_count(full_review,pattern="[A-Za-z,;'\"\\s]+[^.!?]*[.?!]"))%>%
  summarize_at(c('characters','words','sentences'),.funs = mean,na.rm=T)

r_characters = cor.test(nchar(long_review$full_review),long_review$rating_value)
r_words = cor.test(str_count(string = long_review$full_review,pattern = '\\S+'),long_review$rating_value)
r_sentences = cor.test(str_count(string = long_review$full_review,pattern = "[A-Za-z,;'\"\\s]+[^.!?]*[.?!]"),long_review$rating_value)
correlations = data.frame(r = c(r_characters$estimate, r_words$estimate, r_sentences$estimate),p_value=c(r_characters$p.value, r_words$p.value, r_sentences$p.value))
rownames(correlations) = c('Characters','Words','Sentences')
correlations
```

The initial analysis indicated that while the syntax of reviews was significantly related to the review ratings, the correlation was minimal, suggesting that while the way reviews are written matters, it doesn't strongly predict the rating. Despite being statistically significant, the strength of the relationship between syntax and ratings is weak. In practical terms, while changes in syntax are related to changes in ratings, this relationship does not explain much about the rating variance. This may be because even if a review is well structured, the content of the review has little correlation or relevance with the syntax of the review.

## **Variable Description**

|                |                                              |               |                                                     |
|----------------|----------------------------------------------|---------------|-----------------------------------------------------|
| **Variable**   | **Description**                              | **Variable**  | **Description**                                     |
| title          | title of the movie                           | averageRating | weighted average of all the individual user ratings |
| titleId        | alphanumeric unique identifier of the title  | numVotes      | number of votes the title has received              |
| region         | region for this version of the title         | directors     | director(s) of the given title                      |
| language       | language of the title                        |               |                                                     |
| startYear      | release year of a title                      |               |                                                     |
| runtimeMinutes | primary runtime of the title, in minutes     |               |                                                     |
| genres         | up to three genres associated with the title |               |                                                     |

## Analyses

### Bing Sentiment Analysis

We leveraged the Bing lexicon within the tidytext library to carry out the initial sentiment analysis using the \"bag of words\" approach, to divide the text into positive or negative categories.

```{r}
#Tokenize the Data
library(tidytext)
tokenized_df = 
  long_review%>%
  unnest_tokens(input = full_review, output = word)%>%
  select(word)%>%
  anti_join(stop_words)%>%
  group_by(word)%>%
  summarize(count = n())%>%
  ungroup()%>%
  arrange(desc(count))%>%
  top_n(25)
tokenized_df

as.data.frame(get_sentiments('bing'))[1:50,]
long_review%>%
  group_by(data_review_id)%>%
  unnest_tokens(output = word, input = full_review)%>%
  inner_join(get_sentiments('bing'))%>%
  group_by(sentiment)

library(ggplot2)
long_review%>%
  select(data_review_id,full_review,rating_value)%>%
  group_by(data_review_id, rating_value)%>%
  unnest_tokens(output=word,input=full_review)%>%
  ungroup()%>%
  inner_join(get_sentiments('bing'))%>%
  group_by(rating_value,sentiment)%>%
  summarize(n = n())%>%
  mutate(proportion = n/sum(n))%>%
  ggplot(aes(x=rating_value,y=proportion,fill=sentiment))+
  geom_col()+
  coord_flip()

imdb_bing = long_review%>%
  group_by(data_review_id, rating_value)%>%
  unnest_tokens(output = word, input = full_review)%>%
  inner_join(get_sentiments('bing'))%>%
  group_by(data_review_id,rating_value)%>%
  summarize(positive_words = sum(sentiment=='positive'),
            negative_words = sum(sentiment=='negative'),
            proportion_positive = positive_words/(positive_words+negative_words))%>%
  ungroup()

#imput missing data
library(mice)
complete_imdb = mice::complete(mice(imdb_bing,seed = 617))
imdb = mice::complete(mice(long_review,seed = 617))

#Correlation betweeen positive words and rating value (relatively strong and positive)
cor(complete_imdb$proportion_positive, complete_imdb$rating_value)
```

A correlation of 0.5184615 was observed between the count of positive words in a review and the review rating, supporting the intuitive connection that more positive language typically corresponds to higher ratings. This relationship was visualized in a graph within the R environment. The moderate, positive correlation means that as the number of positive words increases, the review ratings tend to increase as well. However, the relationlationship is not perfect or extremely strong.

### Afinn Sentiment Analysis

Similarly, we leveraged the afinn lexicon to assign a numeric value to the emotion pulled from the tokenized text in the reviews. Thus this lexicon can be thought of as computing a sentiment score for words.

```{r}
afinn = as.data.frame(get_sentiments('afinn'))[1:50,]

sentiment_review = 
  imdb %>%
  select(id,full_review,rating_value)%>%
  filter(rating_value != '')%>%
  group_by(id, rating_value)%>%
  unnest_tokens(output=word,input=full_review)%>%
  inner_join(afinn)%>%
  summarize(sentiment_score = mean(value))

sentiment_review

# Assuming 'rating_value' might be a character or factor and needs to be in numerical form
sentiment_review$rating_value <- as.numeric(as.character(sentiment_review$rating_value))

#Plot Sentiment Score vs. Rating Value
library(ggplot2)
ggplot(sentiment_review, aes(x = rating_value, y = sentiment_score)) +
  geom_point() +
  geom_smooth(method = lm, color = "blue") +  # Adds a regression line
  labs(x = "Rating Value", y = "Sentiment Score", title = "Correlation between Rating and Sentiment Score") +
  theme_minimal()

# Calculating Pearson correlation
correlation_result = cor(sentiment_review$rating_value, sentiment_review$sentiment_score, use = "complete.obs")
correlation_result
```

The Afinn sentiment analysis scored poorly with a correlation of 0.03267549 with review ratings, making it an unsuitable choice for sentiment analysis within this dataset due to its low predictive value. This low correlation implies that the sentiment categorized by Afinn does not capture the nuances that influence the viewers ratings of a movie. In this given case, the Afinn method may not have been sufficient enough to capture the subjective interpretation of film reviews, thereby not proving impactful in predicting movie ratings.

### Topic Models and LSA

Prior to running our predictive models, we carried out the following process:

1.  **Corpus Creation and Text Cleaning:** A corpus is constructed from the 'full_review' column, which undergoes text preprocessing to cleanse and normalize the data.

2.  **Dictionary Construction:** From the Document-Term Matrix (DTM) of the cleaned corpus, a dictionary of frequently occurring terms is created, highlighting the most common elements of the reviews.

3.  **DTM Creation and Stem Preprocessing:** A refined DTM is developed from the preprocessed corpus. During this phase, infrequent terms are discarded, and a stem completion technique is applied to the remaining terms to ensure consistency and reduce redundancy.

4.  **TF-IDF Computation and Visualization:** The Term Frequency-Inverse Document Frequency (TF-IDF) metric is calculated to evaluate the importance of terms within the corpus, and a visualization is created to display the top 20 significant terms.

**DTM and IMDb Data Integration:** The final step involves merging the DTM with the broader IMDb dataset, enriching the existing data with detailed textual insights from the reviews, potentially enabling deeper analysis and understanding of user sentiments and topics discussed.

```{r}
#Create corpus and clean the text
library(tm)
corpus = Corpus(VectorSource(imdb$full_review))
corpus = tm_map(corpus,FUN = content_transformer(tolower))
corpus = tm_map(corpus,FUN = removeWords,c(stopwords("en"), "movie", "film"))
corpus = tm_map(corpus,FUN = removePunctuation)
corpus = tm_map(corpus,FUN = stripWhitespace)

#create a dictionary
dict = findFreqTerms(DocumentTermMatrix(Corpus(VectorSource(imdb$full_review))),
                     lowfreq = 0)
dict_corpus = Corpus(VectorSource(dict))

#Create a stem document
corpus = tm_map(corpus,FUN = stemDocument)
dtm = DocumentTermMatrix(corpus)

xdtm = removeSparseTerms(dtm,sparse = 0.95)
xdtm

#Complete the stems created
xdtm = as.data.frame(as.matrix(xdtm))
colnames(xdtm) = stemCompletion(x = colnames(xdtm),
                                dictionary = dict_corpus,
                                type='prevalent')
colnames(xdtm) = make.names(colnames(xdtm))

sort(colSums(xdtm),decreasing = T)


dtm_tfidf = DocumentTermMatrix(x=corpus,
                               control = list(weighting=function(x) weightTfIdf(x,normalize=F)))
xdtm_tfidf = removeSparseTerms(dtm_tfidf,sparse = 0.95)
xdtm_tfidf = as.data.frame(as.matrix(xdtm_tfidf))
colnames(xdtm_tfidf) = stemCompletion(x = colnames(xdtm_tfidf),
                                      dictionary = dict_corpus,
                                      type='prevalent')
colnames(xdtm_tfidf) = make.names(colnames(xdtm_tfidf))
sort(colSums(xdtm_tfidf),decreasing = T)

library(tidyr); library(dplyr); library(ggplot2); library(ggthemes)
data.frame(term = colnames(xdtm),tf = colMeans(xdtm), tfidf = colMeans(xdtm_tfidf))%>%
  arrange(desc(tf))%>%
  top_n(20)%>%
  gather(key=weighting_method,value=weight,2:3)%>%
  ggplot(aes(x=term,y=weight,fill=weighting_method))+
  geom_col(position='dodge')+
  coord_flip()+
  theme_economist()

imdb_data = cbind(review_rating = imdb$rating_value, xdtm)
imdb_data_xdtm = cbind(review_rating = imdb$rating_value, titleId = imdb$titleId, xdtm)
imdb_data_tfidf = cbind(review_rating = imdb$rating_value,xdtm_tfidf)
```

We conducted a LSA topic model to group words together in order to identify if this had a higher ability to predict ratings. The LSA is a technique used in natural language processing to extract and and represent the contextual meaning of words by grouping similar words into topics. LSA helps identify patterns in word usage across corpus by constructing term-document associations matrix and then reducing its dimensionality. The LSA Topic Models identified various keywords, which were displayed in a graph. However, given this model\'s higher RMSEs, they were not selected for the final model.   Despite the LSA Topic model\'s ability to identify topics within reviews, the model did not accurately identify topics with meaningful relationships with the ratings compared to other approaches outlined in this paper.

## Creating Predictive Models

### Predictive Models for sentiment analysis

We split the dataset into train and test data and created a linear model and rpart tree using all variables against the reviews\' ratings arriving at the RMSEs using the predict function.

-   **Regression Tree:** We built a regression tree using the review rating variable as the response and all other variables in the tran dataset as predictors. 

-   **Linear Regression:** We fit a linear regression model using review rating as the dependent variable and all other variables in the train dataset as independent variables.

Due to vector memory issues with linear regression, despite its low RMSE, the decision was made to proceed with a regression tree model. This model highlighted the importance of variables such as the primary director, the primary genre, certain descriptive words, the runtime, and the number of votes.

```{r}
#Split the Data
set.seed(1606)
split = sample(1:nrow(imdb_data),size = 0.7*nrow(imdb_data))
train = imdb_data[split,]
test = imdb_data[-split,]

#Regression Tree
library(rpart); library(rpart.plot)
tree = rpart(review_rating~.,train)
rpart.plot(tree)

pred_tree = predict(tree,newdata=test)
rmse_tree = sqrt(mean((pred_tree - test$review_rating)^2)); rmse_tree

#Linear Regression
reg = lm(review_rating~.,train)
summary(reg)
pred_reg = predict(reg, newdata=test)
rmse_reg = sqrt(mean((pred_reg-test$review_rating)^2)); rmse_reg

xdtm_topic = xdtm[which(rowSums(xdtm)!=0),]

library(topicmodels)
set.seed(1606)
topic2 = LDA(x = xdtm_topic,k = 2)
terms(topic2,10)

df_beta = data.frame(t(exp(topic2@beta)),row.names = topic2@terms)
colnames(df_beta) = c('topic1','topic2')
df_beta[1:20,]

#Topics that differ
topic2%>%
  tidy(matrix='beta')%>%
  mutate(topic = paste0("topic_",topic))%>%
  spread(key = topic,value = beta)%>%
  mutate(beta_spread = log(topic_1/topic_2,2))%>%
  filter(beta_spread>mean(beta_spread)+2*sd(beta_spread)|beta_spread<mean(beta_spread)-2*sd(beta_spread))%>%
  ggplot(aes(x=reorder(term, beta_spread),y=beta_spread, fill=factor(beta_spread>0)))+
  geom_col()+
  coord_flip()+guides(fill=F)+xlab('')

#Join the topics probability with the ratings
text_topics = cbind(as.integer(topic2@documents),topic2@gamma)
colnames(text_topics) = c('id','topic1','topic2')
text_topics = merge(x = text_topics,y = imdb[,c(1,4)],by=c('id','id'))
head(text_topics)

set.seed(1606)
split = sample(1:nrow(text_topics),size = 0.7*nrow(text_topics))
train = text_topics[split,]
test = text_topics[-split,]

model = rpart(review_rating~.-id,train)
pred = predict(model,newdata = test)
topic_model_rmse = sqrt(mean((pred-test$review_rating)^2))

#LSA Model
library(lsa)
clusters = lsa(xdtm)
# lsa decomposes data into three matrices. The term matrix contains the dimensions from svd
clusters$tk = as.data.frame(clusters$tk)
colnames(clusters$tk) = paste0("dim",1:51)
head(clusters$tk)

clusters_data = cbind(id = imdb$id, review_rating = imdb$rating_value,clusters$tk[,53])
clusters_data = as.data.frame(clusters_data)

set.seed(617)
split = sample(1:nrow(clusters_data),size = 0.7*nrow(clusters_data))
train = clusters_data[split,]
test = clusters_data[-split,]

model = rpart(review_rating~.-id,train)
pred = predict(model,newdata = test)
lsa_rmse = sqrt(mean((pred-test$review_rating)^2))
```

A linear regression model and a regression tree model were constructed based on the Bing sentiment analysis. The linear regression model had the lowest Root Mean Square Error (RMSE) of 2.658657, while the regression tree model was slightly less accurate with an RMSE of 2.833017. LSA Topic Models and another topic model yielded higher RMSEs of 2.961321 and 2.959015, respectively, indicating lower prediction accuracy.

#### Feature Selection

Due to vector memory issues with linear regression, despite its low RMSE, the decision was made to proceed with a regression tree model for the final analysis. This model highlighted the importance of variables such as the primary director, the primary genre, which are are often associated with a movie\'s success; certain descriptive words, indicating that the language used in the reviews correlates with the ratings movies receive; the runtime, and the number of votes (numVotes), which speak to the movie\'s length and audience engagement indicating that these variables also have an influence on the overall rating.

```{r}
#Join the imdb dataset to the other factors we've found from the official IMDB API
movie_info = read.csv("/Users/anusha/Downloads/clean_dataset.csv")
# Having issues with building a model 
#Join the movie information with the parsed reviews
imdb_join = merge(movie_info, imdb_data_xdtm, by = "titleId")

#Impute missing values took too much memory and due to the large nature of the dataset we will simply remove observations that don't have a rating_value. 
#library(mice)
#imdb_join_imp <- mice(imdb_join, method = 'bagImpute', m = 1, maxit = 5, seed = 1606)
#imdb_join_complete <- complete(imdb_join_imp)

columns_to_remove <- grep("^director_[4-9]$|^director_[1-5][0-9]$|^director_6[0-6]$", names(imdb_join))

# Remove the matched columns
imdb_join <- imdb_join[ , -columns_to_remove]

#Remove NA rating_values
imdb_join = subset(imdb_join, !is.na(averageRating))
imdb_join$genre_1 <- as.factor(imdb_join$genre_1)
imdb_join$director_1 <- as.factor(imdb_join$director_1)
imdb_join$titleId <- as.factor(imdb_join$titleId)
imdb_join <- imdb_join[, !(names(imdb_join) %in% c("title", "region", "language", "genres", "directors", "director_3", "director_2", "genre_3", "genre_2", "startYear", "average_rating"))]
mean_runtime = mean(imdb_join$runtimeMinutes, na.rm = TRUE)
imdb_join$runtimeMinutes[is.na(imdb_join$runtimeMinutes)] = mean_runtime
imdb_join$runtimeMinutes <- as.numeric(as.character(imdb_join$runtimeMinutes))

set.seed(1606)
split = createDataPartition(y=imdb_join$averageRating,p = 0.7,list = FALSE,groups = 100)
train = imdb_join[split,]
test = imdb_join[-split,]

#Correlation Matrix
imdb_important = imdb_join[, (names(imdb_join) %in% c("review_rating","numVotes", "runtimeMinutes", "bad", "best", "life", "act", "scene", "music", "thing"))]
library(ggcorrplot)
ggcorrplot(cor(imdb_important),
           method = 'square',
           type = 'lower',
           show.diag = F,
           colors = c('#d7191c', '#ffffbf', '#1a9641'))
cor(imdb_important)

#Regression Tree
library(rpart); library(rpart.plot)
tree = rpart(review_rating~.,data=train)
rpart.plot(tree)
summary(tree)
pred_tree = predict(tree,newdata=test)
rmse_tree = sqrt(mean((pred_tree - test$review_rating)^2)); rmse_tree

tree$variable.importance
important_variables = c("director_1", "genre_1", "bad", "runtimeMinutes", "numVotes", "noth", "scene", "act")

formula_str = paste("review_rating ~", paste(important_variables, collapse = " + "))
formula = as.formula(formula_str)

# Now use this formula in the rpart function
tree_refined = rpart(formula, data = train)
rpart.plot(tree_refined)
summary(tree_refined)
pred_tree_refined = predict(tree_refined,newdata=test)
rmse_tree_refined = sqrt(mean((pred_tree_refined - test$review_rating)^2))
rmse_tree_refined


#Linear Regression is causing a vector memory exhausted (limit reached?) error which is suggesting that using a regression tree is a better option
#Espeially since many of the variables are categorical we chose a regression tree
reg_full = lm(review_rating ~ ., data = train)
summary(reg_full)
pred_reg_full = predict(reg_full, newdata = test)
rmse_reg_full = sqrt(mean((pred_reg_full - test$review_rating)^2))
```

The refined regression tree not only yielded the lowest RMSE at 2.595715 but also confirmed the significant impact of the director and genre on ratings. Moreover, specific words were deemed important, suggesting avenues for future research. Interestingly, runtime and the number of votes (numVotes), which were not part of the original hypothesis, also showed significant effects on the final rating, highlighting their unexpected predictive value. The variable importance and correlation matrix supporting these findings were detailed in the R code analysis.

### Director Demographics Analysis

In the pursuit of understanding the indicators influencing film ratings, this report expands upon the examination of the directors\' impact by incorporating demographic variables, specifically race and gender. The categorization was split into 'White' and 'POC' (People of Color) to facilitate a focused analysis while mitigating the potential confounding effects of a more granular racial classification. For gender, a binary classification was employed, distinguishing directors as 'Male' or 'Female'.

To ascertain the pertinent directors for this segment of the analysis, it was necessary to integrate an additional dataset, 'name.basics.tsv.gz', sourced from the IMDb  non-commercial datasets. This dataset was instrumental in correlating the directors' names with their respective unique identifiers previously identified in our datasets. The inherent limitation of our data sources was the absence of explicit information on the directors' gender or race.

Addressing this limitation required a manual approach. We selected directors who represented the extremes of our dataset---those whose films had the highest and lowest average ratings, respectively. This selection yielded a sample of 60 directors comprising the top 30 and bottom 30 based on their average movie ratings.

The manual data entry process involved attributing race and gender to these directors, recording the information within two newly established columns: 'white_or_poc' and 'gender'. This preparatory step was crucial for advancing our statistical analyses.

However, among the race categories of white and poc, we noticed a more apparent disparity that directors falling within the poc category rank higher among the best directors. While for the white category, we noticed an opposite trend.

```{r}
library(dplyr)
library(tidyr)

#Read the clean dataset again for directors' info
director_info = read.csv("/Users/anusha/Desktop/Columbia School/frameworks2/Project/clean_dataset.csv")

#Read the names dataset taken from imdb database to match director IDs to their names
names = read_tsv("/Users/anusha/Downloads/name.basics.tsvv", stringsAsFactors = FALSE)

# Create a named vector with director IDs as names and primary names as values
name_mapping = setNames(names$primaryName, names$nconst)

# Filtering movies from the year 2000 onwards
director_info1 <- director_info[director_info$startYear >= 2000, ]

# Filtering movies to be in english
director_info2 <- director_info1[director_info1$language == "en", ]

# Filtering movies to have number of votes >2000 (for credibility)
director_info3 <- director_info2[director_info2$numVotes >= 3000, ]

# Normalize director IDs from the 'directors' column and replace them with names
director_info4 <- director_info3 %>%
  mutate(directors = strsplit(as.character(directors), ",")) %>%
  unnest(directors) %>%
  mutate(directors = trimws(directors), # Remove any leading/trailing whitespace
         director_name = name_mapping[directors]) %>%
  select(-directors) %>%
  group_by(director_name) %>%
  summarize(averageRating = mean(as.numeric(as.character(averageRating)), na.rm = TRUE),
            numMovies = n(), .groups = 'drop')

# Get the top 30 and lowest 30 directors with at least 5 movies directed
best_directors <- director_info4 %>%
  filter(numMovies > 5) %>%
  arrange(desc(averageRating)) %>%
  slice_head(n = 30)

worst_directors <- director_info4 %>%
  filter(numMovies > 5) %>%
  arrange(averageRating) %>%
  slice_head(n = 30)

# Print the results
print("Top 30 Directors:")
print(best_directors)
print("Lowest 30 Directors:")
print(worst_directors)

# Load required libraries
library(dplyr)
library(ggplot2)

# After manually adding demographic data to the previously generated datasets, we will read them back into R
best_directors_demo <- read.csv("/Users/anusha/Downloads/best_directors_demo.csv", stringsAsFactors = FALSE)
worst_directors_demo <- read.csv("/Users/anusha/Downloads/worst_directors_demo.csv", stringsAsFactors = FALSE)

# Descriptive Statistics
summary(best_directors_demo)
summary(worst_directors_demo)

# Add a 'group' column manually
best_directors_demo$group <- 'Best'
worst_directors_demo$group <- 'Worst'

# Combine the datasets
combined_directors <- rbind(best_directors_demo, worst_directors_demo)

# Visualizations
# Comparing Gender Distribution
ggplot(data = combined_directors, aes(x = gender, fill = group)) +
  geom_bar(position = "dodge") +
  labs(title = "Gender Distribution among Best and Worst Directors", x = "Gender", y = "Count")

# Comparing Race Distribution
ggplot(data = combined_directors, aes(x = white_or_poc, fill = group)) +
  geom_bar(position = "dodge") +
  labs(title = "White vs. POC Distribution among Best and Worst Directors", x = "White_or_POC", y = "Count")

# Comparative Analysis using Chi-Square Test for Categorical Data
# Gender
gender_table <- table(combined_directors$group, combined_directors$gender)
chisq_gender = chisq.test(gender_table)
chisq_gender

# Race
race_table <- table(combined_directors$group, combined_directors$white_or_poc)
chisq_race = chisq.test(race_table)
chisq_race

# Correlation Analysis
# Checking correlation between 'averageRating' and 'numMovies'

cor(best_directors_demo$averageRating, best_directors_demo$numMovies, use = "complete.obs")
cor(worst_directors_demo$averageRating, worst_directors_demo$numMovies, use = "complete.obs")

# Regression Analysis
# Predicting rating based on demographics and number of movies
best_lm <- lm(averageRating ~ gender + white_or_poc + numMovies, data = best_directors_demo)
summary(best_lm)
worst_lm <- lm(averageRating ~ gender + white_or_poc + numMovies, data = worst_directors_demo)
summary(worst_lm)

```

#### **Comparative Analysis using Chi-Square Test for Categorical Data**

**Gender**

Given the X-squared value of 0 and a p-value of 1, the test indicates that there is no statistical evidence to suggest a difference in the gender distribution between the top and bottom directors based on the data provided. This means that, as far as the test is concerned, gender is similarly distributed between the best and worst directors in your dataset.

**Race**

The low p-value suggests that there is a statistically significant association between the directors' race and their classification as either best or worst directors based on your dataset. This means that race appears to be a factor in how directors are distributed between these two categories---there are more of one race in one group compared to what would be expected if the distribution were random.\
\
Note: Given this result, it might be inferred that systemic factors, cultural influences, or other unmeasured variables could be contributing to this disparity. This finding could warrant a deeper examination into how race influences or correlates with the success or challenges faced by directors in the film industry.

#### **Correlation Analysis between Average Rating and number of movies**

**Best directors:** This indicates a weak to moderate negative correlation between the number of movies directed and the average rating among the best directors. It suggests that, within this group, directors with more movies tend to have slightly lower average ratings. This might imply that as directors take on more projects, there could be a slight dilution in the quality or perceived quality of their work, or it could be that more experienced directors have varied receptions to their films.

**Worst directors:** This also indicates a weak negative correlation, but it is less pronounced than with the best directors. Among the worst directors, those who have directed more movies also tend to have slightly lower average ratings. This could suggest that continuous production does not necessarily lead to improvements in ratings, or that these directors might be consistently producing lower-rated films.

#### **Regression Analysis to predict rating based on demographics and number of movies**

**Best directors:** A correlation of -0.3026625 was observed among the best directors between demographics and number of movies, the lack of significant predictors in the regression model for the best directors implies that other factors not included in the model may be more influential in determining their ratings. These could include the specific content of the movies, genre preferences, production values, or external market factors. It underscores the complexity of movie success and suggests that demographic factors alone do not drive ratings for the best directors.\

**Worst directors:** A correlation of -0.193824 was observed among the worst directors between demographics and number of movies, the model\'s inability to find significant predictors also points to the likelihood that other unexamined factors may be influencing their ratings. These might include the nature of the projects they choose, their market positioning, or potentially the budgets and marketing support their films receive. The analysis suggests that simply increasing movie output does not improve ratings among directors who are already performing poorly, and demographic traits like gender and race do not appear to play a role in their ratings outcomes.

## Conclusion

\
We noticed that the Director's identity and movie\'s genre significantly impacted ratings. Audience engagement (i.e. numVotes) and quality elements (i.e. \"scene\", \"music\") also play vital roles in determining movie ratings. Therefore, directorial talent, genre selection, audience engagement, and the quality of various movie elements collectively shape audience perceptions and ratings, influencing the success of a film. 

In relation to our variables, the number of votes a movie receives (numVotes) often correlates with its popularity and the level of interest it generates. Higher number of votes, and hence higher engagement, can indicate a wider reach and stronger reactions from the viewing public. This metric can be crucial for understanding how broadly appealing and \"successful\" a movie is. Quality elements like \"scene\" and \"music\" might not be directly related to the director's identity or movie\'s genre, but they also play a crucial role in a film\'s ratings. Scene elements like cinematography and set design are important elements for visual storytelling. Well-crafted scenes can enhance the storytelling. Furthermore, a film\'s score and soundtrack significantly affect the emotional tone of the movie. Effective use of music can heighten drama, accentuate emotional moments, and even become iconic elements that define the film. Hence, making the movie more engaging through effective use of scene elements and music score may affect user ratings for the movie. 

Various factors influence how a movie is perceived by its audience. High-quality execution in these areas can lead to higher ratings and hence greater success as defined in this paper, while poor execution can do the opposite. The synergy between a talented director, the right genre choice, high audience engagement, and excellent execution of technical and artistic elements is what typically defines the most successful, or highly rated films. Each factor not only stands alone but also interacts with the others, collectively shaping the viewers\' experience and the ratings. 

## **Future Recommendations**

At the moment the project is focused on conducting a predictive model using sentiment analysis to identify disparities amongst movies. However, there is a potential extension we\'d like to consider should time permit. Another way to evaluate our hypothesis would be to collect movies with the same ratings. Then reverse the research method by giving each model a sentiment analysis-based score. In doing so we can cluster the scores and identify the attributes shared amongst the clusters. This would help identify the disparities in the sentiments of movies and hopefully refine the model to intake important variables such as genre or director. 

Furthermore, we constantly encountered issues where the error read \"vector memory exhausted\" which acted as a large obstacle to our analysis. This is a major reason why we couldn\'t consider secondary genres and directors in our analysis and including these would help improve the accuracy of our research project. In the future with more skill and cleaner data we could circumvent the problem through creating dummy variables. 

Lastly, we found it incredibly hard to develop an analysis that considered director demographics because the dataset was very difficult to find. One possible solution would be to look up gender or race lexicons to join with our dataset. With looser time constraints, ensuring to find a comprehensive dataset containing director demographics or scraping this from IMDB\'s unofficial API would be an extension and re-evaluation of the findings of this study. An example of this would be to use the baby names directory posted by the Social Security Administration which includes names of babies born in the US and their gender.

By exploring further avenues like these, we can enhance our understanding of the demographic landscape in film direction, which is crucial for addressing diversity and representation issues within the industry. This would not only enrich our study but also contribute to broader discussions and efforts towards inclusivity in cinema.

## Code Appendix

Initially we considered doing this analysis on the short review but quickly realized that creating a dictionary without stop words with significance was very difficult.

```{r}
tokenized_df = 
  long_review%>%
  unnest_tokens(input = short_review, output = word)%>%
  select(word)%>%
  anti_join(stop_words)%>%
  group_by(word)%>%
  summarize(count = n())%>%
  ungroup()%>%
  arrange(desc(count))%>%
  top_n(25)
tokenized_df

library(tm)
corpus = Corpus(VectorSource(imdb$short_review))
corpus = tm_map(corpus,FUN = content_transformer(tolower))
corpus = tm_map(corpus,FUN = removeWords,c(stopwords("en"), "movie", "film"))
corpus = tm_map(corpus,FUN = removePunctuation)
corpus = tm_map(corpus,FUN = stripWhitespace)

#create a dictionary
dict = findFreqTerms(DocumentTermMatrix(Corpus(VectorSource(imdb$short_review))),
                     lowfreq = 0)
dict_corpus = Corpus(VectorSource(dict))

#Create a stem document
corpus = tm_map(corpus,FUN = stemDocument)
dtm = DocumentTermMatrix(corpus)

xdtm = removeSparseTerms(dtm,sparse = 0.95)
xdtm

xdtm = as.data.frame(as.matrix(xdtm))
colnames(xdtm) = stemCompletion(x = colnames(xdtm),
                                dictionary = dict_corpus,
                                type='prevalent')
colnames(xdtm) = make.names(colnames(xdtm))

sort(colSums(xdtm),decreasing = T)


dtm_tfidf = DocumentTermMatrix(x=corpus,
                               control = list(weighting=function(x) weightTfIdf(x,normalize=F)))
xdtm_tfidf = removeSparseTerms(dtm_tfidf,sparse = 0.95)
xdtm_tfidf = as.data.frame(as.matrix(xdtm_tfidf))
colnames(xdtm_tfidf) = stemCompletion(x = colnames(xdtm_tfidf),
                                      dictionary = dict_corpus,
                                      type='prevalent')
colnames(xdtm_tfidf) = make.names(colnames(xdtm_tfidf))
sort(colSums(xdtm_tfidf),decreasing = T)
```

The output of this analysis was only the term good which we considered a less than ideal term collection to conduct any analysis.

## References

Basuroy, S., Desai, K. K., & Talukdar, D. (2006). An empirical investigation of signaling in the 		motion picture industry. Journal of Marketing Research, 43(2), 287-295. 		

https://doi.org/10.1509/jmkr.43.2.287\
\
Boatwright, P., Basuroy, S., & Kamakura, W. (2007). Reviewing the reviewers: The impact of 

	individual film critics on box office performance. Quantitative Marketing and 

Economics, 5(4), 401-425. <https://doi.org/10.1007/s11129-007-9029-1>

Loupos, P., Peng, Y., Li, S., & Hao, H. (2023). What reviews foretell about opening weekend box 

office revenue: The harbinger of failure effect in the movie industry. Marketing Letters, 

34(3), 513-534. <https://doi.org/10.1007/s11002-023-09665-8>\

Statista Research Department. (2024, January 3). Box office revenue in the U.S. and Canada 

2023\. Statista. https://www.statista.com/statistics/187069/north-american-box-office-gross-revenue-since-1980/
