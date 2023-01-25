# Drop the Latitude and Longitude columns from volunteer
volunteer_cols = volunteer.drop(["Latitude", "Longitude"], axis=1)

# Drop rows with missing category_desc values from volunteer_cols
volunteer_subset = volunteer_cols.dropna(subset=["category_desc"])

# Print out the shape of the subset
print(volunteer_subset.shape)

-------------
# Print the head of the hits column
print(volunteer["hits"].head())

# Convert the hits column to type int
volunteer["hits"] = volunteer["hits"].astype("int")

# Look at the dtypes of the dataset
print(volunteer.dtypes)

---------------------
# Create a data with all columns except category_desc
X = volunteer.drop("category_desc", axis=1)

# Create a category_desc labels dataset
y = volunteer[["category_desc"]]
# Use stratified sampling to split up the dataset according to the y dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y ,random_state=42)

# Print the category_desc counts from y_train
print(y_train['category_desc'].value_counts())
