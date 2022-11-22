#!/usr/bin/env python
# coding: utf-8

# # Project: Investigating the TMBD Movie Dataset
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# > This data set contains information of about 10,000 movies collected from The Movie Database (TMDb).
# 
# >There are 21 columns in this dataset -
# 
#        id - Movie Identification Number
#        imdb_id - IMDB Identification Number
#        popularity - Movie popularity score
#        budget - Amount spent on movie
#        revenue - Amount generated from movie
#        original_title - Movie title
#        cast - Actors in movie
#        homepage - Homepage
#        director - Movie directors
#        tagline - Tagline
#        keywords - Keywords
#        overview - Overview
#        runtime - Duration of movie in minutes
#        genres - Movie genre
#        production_companies - Production companies
#        release_date - Movie release date (dd-mm-yyyy)
#        vote_count - Number of ratings for movie
#        vote_average - Average rating for movie
#        release_year - Movie release year (yyyy)
#        budget_adj - Adjusted amount spent on movie
#        revenue_adj - Adjusted amout generated from movie
#        
# >In this project, the following will be looked into; 
# 
#        How the number of movies released per year has changed over the years.
# 
#        How the vote count has changed over the years.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 
# 
# ### General Properties

# In[37]:


#load data and display the first 5 and last 5 rows of data
movie_df = pd.read_csv('tmdb-movies.csv')
print(movie_df.head())
print(movie_df.tail())


# In[26]:


#check number of rows and columns in data
movie_df.shape


# #### There are a total of 10,866 rows and 21 columns in the TMDB movie dataset

# In[27]:


movie_df.info()


# #### The table headers are labelled properly and do not need any modifications.
# #### There are 9 columns with missing values.
# #### There are 10 missing imdb_id values, imdb_id is a unique identifier so rows with missing imdb_id will be dropped in order to have a more accurate dataset.
# #### The other coulmns with missing values will not be dropped but instead null values will be replaced with 'Not Available' to not only give a cleaner dataset but to also prevent our int values from being presented as float.
# #### The Dtype for release_date is currently in object and will be changed to datetime.
# #### The Dtype for budget_adj and revenue_adj are currently in float64 and will be changed to int.

# In[28]:


movie_df.describe()


# #### The dataset contains information of movies released between 1960 and 2015.
# #### Although there were no missing values for the buget, revenue, budget_adj and revenue_adj columns, there are rows with 0 values for these columns. I will perform a count on the rows where values are greater than 0 to determine whether these rows can be used in my EDA.

# In[29]:


#check for rows with non-zero values
columns = ['budget','revenue','budget_adj','revenue_adj']
def non_zero_values():
    for col in columns:
        print(movie_df[movie_df[col] > 0].shape)
        
non_zero_values()


# #### For the budget and budget_adj columns, 5,170 rows out of 10,866 rows have values greater than 0.
# #### For the revenue and revenue_adj columns, 4,850 rows out of 10,866 rows have values greater than 0.

# In[30]:


#check for duplicate entries
sum(movie_df.duplicated())


# #### There is an instance of a duplicated row

# In[31]:


#check for uniques values 
movie_df.nunique()


# #### Now that the loaded data has been thoroughly checked, I will be proceeding to make the necessary changes highlighted above.

# 
# 
# ### Data Cleaning

# In[32]:


#create copy of dataset
movie_clean = movie_df.copy()


# In[38]:


# dropping rows with missing imdb_id
movie_clean = movie_clean[movie_clean['imdb_id'].notnull()]


# In[39]:


# checking that the rows with missing imdb_id have been dropped
print(movie_clean.info())
print(movie_clean.shape)


# #### The rows with missing imdb_id have been dropped and the movie dataframe now has 10,856 entries(rows).

# In[40]:


# filling other columns with missing values
movie_clean = movie_clean.fillna('Not Available',inplace=False)
# this code was obtained from the pandas documentation file - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html


# In[41]:


# checking that missing values have been filled 
movie_clean.info()


# #### Missing values have been replaced with 'Not Available'

# In[50]:


#Changing release_date Dtype to datetime from object
movie_clean['release_date'] = pd.to_datetime(movie_clean['release_date'])
print(movie_clean['release_date'].dtypes)


# In[49]:


#Changing budget_adj and revenue_adj Dtype to int64 from float64
columns = ['budget_adj','revenue_adj']
def change_dtype():
    for col in columns:
        movie_clean[col] = movie_clean[col].astype(np.int64)
        print(movie_clean[col].dtypes)
change_dtype()


# #### All columns are now in the correct Dtype format

# In[52]:


# dropping duplicates in dataframe and confirming they have been dropped
movie_clean.drop_duplicates(inplace=True)
sum(movie_clean.duplicated())


# #### There are currently no duplicate rows in the dataframe

# #### My dataset has been cleaned and I will now be proceeding to run some exploratory data analysis.

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# ### How has the number of movies released per year changed?
# ### How many movies were released in each year, which years had the highest and lowest number of released movies? What months had the highest and lowest number of released movies?

# In[55]:


#create a subset for this EDA
movie_subset = movie_clean.copy()
movie_subset = movie_subset[['id','imdb_id','original_title','genres','release_date','release_year']]


# In[58]:


# number of movies released in each year
movie_subset.groupby(['release_year'])['imdb_id'].count()


# In[59]:


# years with the highest number of released movies - Top 10
movie_subset['release_year'].value_counts().nlargest(10)
#.nlargest() function was extracted from https://stackoverflow.com/questions/35364601/group-by-and-find-top-n-value-counts-pandas


# In[60]:


# a horizontal bar chart showing the top 10 years with number of released movies
movie_subset['release_year'].value_counts().nlargest(10).plot(kind='barh', alpha=.7,color='green', figsize=(10,8));
plt.title('Years with Highest Number of Released Movies', fontsize=16)
plt.xlabel('Number of released movies', fontsize=14);
plt.ylabel('Release year', fontsize=14);


# #### 2014 had the highest number of released movies - 699 movies

# In[61]:


# years with the lowest number of released movies - Bottom 10
movie_subset['release_year'].value_counts().nsmallest(10)


# In[62]:


# a horizontal bar chart showing the bottom 10 years with number of released movies
movie_subset['release_year'].value_counts().nsmallest(10).plot(kind='barh',alpha=.7,color='red',figsize=(10,8));
plt.title('Years with Lowest Number of Released Movies', fontsize=16)
plt.xlabel('Number of released movies', fontsize=14);
plt.ylabel('Release year', fontsize=14);


# #### 1961 and 1969 had the lowest number of released movies - 31 movies

# In[65]:


# number of movies released each month
movie_subset['release_date'].dt.month.value_counts()
# function to extract month from date was extracted from - https://stackoverflow.com/questions/25146121/extracting-just-month-and-year-separately-from-pandas-datetime-column


# In[66]:


# plot to display number of movies released each month
movie_subset['release_date'].dt.month.value_counts().plot(kind='bar',alpha=.7,color='brown',figsize=(10,8));
plt.title('Movies Released Each Month', fontsize=16)
plt.xlabel('Month', fontsize=14);
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11],['Sep','Oct','Dec','Aug','Jan','Jun','Mar','Nov','May','Jul','Apr','Feb'])
plt.ylabel('Number of released movies', fontsize=14);


# #### The month of September has the highest number of released movies - 1,329 while the month of February has the lowest numberof released movies - 691

# ### Has the vote count for movies changed over the years? What years have the best and worst user ratings?  What movies have the best and worst user ratings?

# In[67]:


#create subset for this EDA
movie_subset2 = movie_clean.copy()
movie_subset2 = movie_subset2[['id','imdb_id','original_title','genres','release_date','release_year','vote_count','vote_average']]


# In[69]:


movie_subset2.groupby(['release_year'])['vote_count'].mean().plot(kind='line',alpha=.7, figsize=(15,8), color='green');
plt.title('Average of Vote Count over the years', fontsize=16)
plt.xlabel('Release Year', fontsize=14);
plt.ylabel('Number of Votes', fontsize=14);


# #### The plot above shows that there has been an increase in the number of vote count over the years.

# In[70]:


# displaying user rating distribution
fig, ax = plt.subplots(figsize =(8,6))
ax.hist(movie_subset2['vote_average'], color = 'brown')
ax.set_title('Distributions of User Ratings', fontsize=14)
ax.set_xlabel('Vote Average', fontsize=14)
ax.set_ylabel('Movie Count', fontsize=14)
plt.show();


# #### This distribution shows that a majority of the ratings are within the range of 4.8 - 7.6.
# 
# #### From my initial check of the dataset, there are 72 unique values for user ratings (vote_average column). To get a more detailed visualization of the user ratings for each year, I will be creating a new column to help categorize the user ratings into 4 distinct categories - Excellent, Good, Average and Poor. This will help give a better visualization of the data.

# In[71]:


#Bin edges used to cut the data into groups
bin_edges = [0,2.4,4.9,7.4,9.2]
#Labels for the 4 vote average categories
bin_names = ['poor','average','good','excellent']
#Create new column - ratings category
movie_subset2['ratings_category'] = pd.cut(movie_df['vote_average'],bin_edges,labels=bin_names)
movie_subset2.head()


# #### The ratings_category column has been created.
# #### Movies with an average vote between 7.5 - 9.2 = Excellent
# #### Movies with an average vote between 5.0 - 7.4 = Good
# #### Movies with an average vote between 2.5 - 4.9 = Average
# #### Movies with an average vote between 0 - 2.4 = Poor

# In[72]:


#getting count of movies in each category
movie_subset2['ratings_category'].value_counts()


# In[82]:


#pie chart to display the ratio of movies in each ratings category
movie_subset2['ratings_category'].value_counts().plot(kind='pie',figsize=(8,8),autopct = '% 1.1f %%', explode=(0.1,0,0,0));
plt.title('User Rating Category', fontsize=16);
#code extracted from - https://stackoverflow.com/questions/21090316/plotting-pandas-dataframes-in-to-pie-charts-using-matplotlib


# #### 82.6% of the movies released fall under the Good ratings category
# 
# #### I will do some further analysis to see what years have the best and worst ratings based on vote count and vote average.

# In[100]:


# movies with highest user rating
movie_df.groupby(['original_title'])[['vote_average','release_year']].max().nlargest(20,'vote_average')


# #### 14 out of the top 20 rated movies were released in the 2000s

# In[101]:


# movies with lowest user rating
movie_df.groupby(['original_title'])[['vote_average','release_year']].min().nsmallest(20,'vote_average')


# #### 18 of the worst 20 movies were also released in the 2000s.

# In[83]:


# creating a df to have only movies with an excellent rating
excellent_df = movie_subset2.query('ratings_category == "excellent"')
excellent_df.head()


# In[97]:


# number of excellent movies released in each year
excellent_df.groupby(['release_year'])['release_year'].count().plot(kind='bar',figsize=(15,6),color='green');
plt.title('Best User Ratings', fontsize=16)
plt.xlabel('Release Year', fontsize=14);
plt.ylabel('Number of Movies with Excellent Rating', fontsize=14);
print(excellent_df.groupby(['release_year'])['release_year'].count())


# #### 2014 had the highest number of movies rated Excellent

# In[96]:


# creating a df to have only movies with a poor rating
poor_df = movie_subset2.query('ratings_category == "poor"')
poor_df.head()


# In[99]:


# creating a bar chart for Worst 10 years
poor_df.groupby(['release_year'])['release_year'].count().plot(kind='bar',alpha=.7, figsize=(10,8), color='red');
plt.title('Worst User Ratings', fontsize=16);
plt.xlabel('Release Year', fontsize=14);
plt.ylabel('Number of Movies with Poor Rating', fontsize=14);
print(poor_df.groupby(['release_year'])['release_year'].count())


# #### 2012 is recorded to have the highest number of poor ratings
# 
# #### Considering that more movies were released in the 2000s, I will do some further analysis to determine which years have the worst and best movie ratings

# In[106]:


movie_subset2.groupby(['release_year'])['vote_average'].mean().plot(kind='line',alpha=.7, figsize=(15,8), color='green');
plt.title('Average of User Ratings over the years', fontsize=16)
plt.xlabel('Release Year', fontsize=14);
plt.ylabel('Average User Ratings', fontsize=14);


# #### From the plot above, we can tell that there is a decline in the user ratings with the 1970s having the best user rating.

# <a id='conclusions'></a>
# ## Conclusions

# #### From the analysis done, the following can be concluded:
#    1. The number of movies released per year has increased over the years with about 50% of movies being released after 2005. It is also interesting to note that over the years, the month of September has recorded the highest number of movie releases with 1,329 movies released between 1960 - 2015.
#    2. Although the number of votes has increased over the years, there has been a decline in the average user ratings with the 1970s having the best user ratings. Viewer's favourite in terms of vote average is The Story of Film: An Odyssey while the least favourite movies are Manos: The Hands of Fate and Transmorphers.
# 
# #### A limitation faced in this analysis was the non-viability of the budget and revenue columns, having less than 50% non-zero values. Due to this, I was unable to find a correlation between user ratings and budget and also doing analysis on the profit made on movies over the years. 
