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
