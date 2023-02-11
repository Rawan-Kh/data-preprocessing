# Print the DataFrame info
print(ufo.info())

# Change the type of seconds to float
ufo["seconds"] = ufo["seconds"].astype(float)

# Change the date column to type datetime
ufo["date"] = pd.to_datetime(ufo["date"])

# Check the column types
print(ufo.info())

# Nice job on transforming the column types! This will make feature engineering and standardization much easier.
----------
# Count the missing values in the length_of_time, state, and type columns, in that order
print(ufo[['length_of_time', 'state', 'type']].isna().sum())

# Drop rows where length_of_time, state, or type are missing

# ufo_no_missing = ufo.dropna(ufo[['length_of_time', 'state', 'type']],axis=0)
# right way
ufo_no_missing = ufo.dropna(subset=["length_of_time", "state", "type"])

# Print out the shape of the new dataset
print(ufo_no_missing.shape)

-------------
def return_minutes(time_string):

    # Search for numbers in time_string
    num = re.search(re.compile(r"\d+"), time_string)
    if num is not None:
        return int(num.group(0))
        
# Apply the extraction to the length_of_time column
ufo["minutes"] = ufo["length_of_time"].apply(return_minutes)

# Take a look at the head of both of the columns
print(ufo[['length_of_time','minutes']].head())

# The minutes information is now in a form where it can be inputted into a model.
#        length_of_time     minutes
#     2  about 5 minutes      5.0
#     4       10 minutes     10.0
#     7        2 minutes      2.0
#     8        2 minutes      2.0
#     9        5 minutes      5.0
---------

# Check the variance of the seconds and minutes columns
print(ufo[["seconds", "minutes"]].var())

# Log normalize the seconds column
ufo["seconds_log"] = np.log(ufo["seconds"])

# Print out the variance of just the seconds_log column
print(ufo["seconds_log"].var())

# Now it's time to engineer new features in the ufo dataset.
------------
# Use pandas to encode us values as 1 and others as 0
ufo["country_enc"] = ufo["country"].apply(lambda val:1 if val=="us" else 0)

# Print the number of unique type values
print(len(ufo["type"].unique()))

# Create a one-hot encoded set of the type values
type_set = pd.get_dummies(ufo["type"])

# Concatenate this set back to the ufo DataFrame
ufo = pd.concat([ufo, type_set], axis=1)

---------------
# Look at the first 5 rows of the date column
print(ufo['date'].head())

# Extract the month from the date column
ufo["month"] = ufo["date"].dt.month

# Extract the year from the date column
ufo["year"] = ufo["date"].dt.year

# Take a look at the head of all three columns
print(ufo[['date','month','year']].head())
------------------
# Take a look at the head of the desc field
print(ufo['desc'].head())

# Instantiate the tfidf vectorizer object
vec = TfidfVectorizer()

# Fit and transform desc using vec
desc_tfidf = vec.fit_transform(ufo["desc"])

# Look at the number of columns and rows
print(desc_tfidf.shape)

# You'll notice that the text vector has a large number of columns. (1866, 3422)
# We'll work on selecting the features we want to use for modeling in the next section
------------
# Now to get rid of some of the unnecessary features in the ufo dataset. Because the country column has been encoded as country_enc, you can select it and drop the other columns related to location: city, country, lat, long, and state.

# You've engineered the month and year columns, so you no longer need the date or recorded columns. You also standardized the seconds column as seconds_log, so you can drop seconds and minutes.

# You vectorized desc, so it can be removed. For now you'll keep type.

# You can also get rid of the length_of_time column, which is unnecessary after extracting minutes.
# Make a list of features to drop
to_drop = ['city','country','lat', 'long', 'state','date','desc','date','recorded','seconds','minutes','length_of_time']

# Drop those features
ufo_dropped = ufo.drop(to_drop,axis=1)

# Let's also filter some words out of the text vector we created
filtered_words = words_to_filter(vocab, vec.vocabulary_, desc_tfidf, 4)

# You're almost done. In the next exercises, you'll model the UFO data in a couple of different ways.

---------

# Take a look at the features in the X set of data
print(X.columns)

# Split the X and y sets
X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,random_state=42)

# Fit knn to the training sets
knn.fit(X_train, y_train)

# Print the score of knn on the test sets
print(knn.score( X_test, y_test ))

--------

# you'll build a model using the text vector we created, desc_tfidf,
# using the filtered_words list to create a filtered text vector. 
# Let's see if you can predict the type of the sighting based on the text. 
# You'll use a Naive Bayes model for this.

# Use the list of filtered words we created to filter the text vector
filtered_text = desc_tfidf [:, list(filtered_words)]

# Split the X and y sets using train_test_split, setting stratify=y 
X_train, X_test, y_train, y_test = train_test_split(filtered_text.toarray(), y, stratify=y, random_state=42)

# Fit nb to the training sets
nb.fit(X_train, y_train,)

# Print the score of nb on the test sets
nb.score(X_test, y_test)

#  you've completed the course! As you can see, this model performs very poorly on this text data.  ->  0.17987152034261242
#   This is a clear case where iteration would be necessary to figure out what subset of 
#  text improves the model, and if perhaps any of the other features are useful in predicting type.

