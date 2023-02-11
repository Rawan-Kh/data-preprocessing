# The text field needs to be vectorized before removing it, otherwise we might lose important data.
----------------
# Length, Difficulty, Accessible
# All three of these columns are good candidates for removal during feature selection.
# Prop_ID                     Name                                           Location      Park_Name      Length Difficulty                                      Other_Details Accessible  \
# 0    B057  Salt Marsh Nature Trail  Enter behind the Salt Marsh Nature Center, loc...    Marine Park   0.8 miles       None  <p>The first half of this mile-long trail foll...          Y   
# 1    B073                Lullwater  Enter Park at Lincoln Road and Ocean Avenue en...  Prospect Park    1.0 mile       Easy  Explore the Lullwater to see how nature thrive...          N   
# 2    B073                  Midwood  Enter Park at Lincoln Road and Ocean Avenue en...  Prospect Park  0.75 miles       Easy  Step back in time with a walk through Brooklyn...          N   
# 3    B073                Peninsula  Enter Park at Lincoln Road and Ocean Avenue en...  Prospect Park   0.5 miles       Easy  Discover how the Peninsula has changed over th...          N   
# 4    B073                Waterfall  Enter Park at Lincoln Road and Ocean Avenue en...  Prospect Park   0.5 miles       Easy  Trace the source of the Lake on the Waterfall ...          N   

#   Limited_Access  lat  lon  Length_extract  accessible_enc     Easy  Easy   Easy/Moderate  Moderate  Moderate/Difficult  Various  
# 0              N  NaN  NaN            0.80               1  0     0      0              0         0                   0        0  
# 1              N  NaN  NaN            1.00               0  0     1      0              0         0                   0        0  
# 2              N  NaN  NaN            0.75               0  0     1      0              0         0                   0        0  
# 3              N  NaN  NaN            0.50               0  0     1      0              0         0                   0        0  
# 4              N  NaN  NaN            0.50               0  0     1      0              0         0                   0        0  

------------------
#                                                 title  hits  postalcode  vol_requests_lognorm  created_month  Education  Emergency Preparedness  Environment  Health  Helping Neighbors in Need  \
#     1                                       Web designer    22     10010.0                 0.693              1          0                       0            0       0                          0   
#     2      Urban Adventures - Ice Skating at Lasker Rink    62     10026.0                 2.996              1          0                       0            0       0                          0   
#     3  Fight global hunger and support women farmers ...    14      2114.0                 6.215              1          0                       0            0       0                          0   
#     4                                      Stop 'N' Swap    31     10455.0                 2.708              1          0                       0            1       0                          0   
#     5                               Queens Stop 'N' Swap   135     11372.0                 2.708              1          0                       0            1       0                          0   
    
#        Strengthening Communities  
#     1                          1  
#     2                          1  
#     3                          1  
#     4                          0  
#     5                          0  

# Create a list of redundant column names to drop
to_drop = ["vol_requests","category_desc", "locality", "region", "created_date" ]

# Drop those columns from the dataset
volunteer_subset = volunteer.drop(to_drop, axis=1)

# Print out the head of volunteer_subset
print(volunteer_subset.head())
# It's often easier to collect a list of columns to drop, rather than dropping them individually
---------------

# Print out the column correlations of the wine dataset
print(wine.corr())

# Drop that column from the DataFrame
wine = wine.drop('Flavanoids',axis=1)

print(wine.head())

# Dropping correlated features is often an iterative process, so you may need to try different combinations in your model
--------
# Add in the rest of the arguments
def return_weights(vocab, original_vocab, vector, vector_index, top_n):
    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))
    
    # Transform that zipped dict into a series
    zipped_series = pd.Series({vocab[i]:zipped[i] for i in vector[vector_index].indices})
    
    # Sort the series to pull out the top n weighted words
    zipped_index = zipped_series.sort_values(ascending=False)[:top_n].index
    return [original_vocab[i] for i in zipped_index]

# Print out the weighted words
print(return_weights(vocab, tfidf_vec.vocabulary_, text_tfidf, 8, 3))

-------------
def words_to_filter(vocab, original_vocab, vector, top_n):
    filter_list = []
    for i in range(0, vector.shape[0]):
    
        # Call the return_weights function and extend filter_list
        filtered = return_weights(vocab, original_vocab, vector, i, top_n)
        filter_list.extend(filtered)
        
    # Return the list in a set, so we don't get duplicate word indices
    return set(filter_list)

# Call the function to get the list of word indices
filtered_words = words_to_filter(vocab, tfidf_vec.vocabulary_, text_tfidf, 3)

# Filter the columns in text_tfidf to only those in filtered_words
filtered_text = text_tfidf[:, list(filtered_words)]

# In the next exercise, you'll train a model using the filtered vector.
----------

# Split the dataset according to the class distribution of category_desc
X_train, X_test, y_train, y_test = train_test_split(filtered_text.toarray(), y, stratify=y, random_state=42)

# Fit the model to the training data
nb.fit(X_train , y_train)

# Print out the model's accuracy
print(nb.score(X_test, y_test))

# You can see that our accuracy score wasn't that different from the score at the end of Chapter 3. 
# But don't worry, this is mainly because of how small the title field is
--------------

# Instantiate a PCA object
pca = PCA()

# Define the features and labels from the wine dataset
X = wine.drop("Type", axis=1)
y = wine["Type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Apply PCA to the wine dataset X vector
pca_X_train = pca.fit_transform(X_train)
pca_X_test = pca.transform(X_test)

# Look at the percentage of variance explained by the different components
print(pca.explained_variance_ratio_)

#  In the next exercise, you'll train a model using the PCA-transformed vector
-------------

# Fit knn to the training data
knn.fit(pca_X_train,y_train)

# Score knn on the test data and print it out
knn.score(pca_X_test , y_test)

# PCA turned out to be a good choice for the wine dataset.

