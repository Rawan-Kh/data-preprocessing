# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

knn = KNeighborsClassifier()

# Fit the knn model to the training data
knn.fit(X_train, y_train)

# Score the model on the test data
print(knn.score(X_test,y_test))

# You can see that the accuracy score is pretty low. Let's explore methods to improve this score
----------------

wine.var()
# Out[1]:

# Type                                0.601
# Alcohol                             0.659
# Malic acid                          1.248
# Ash                                 0.075
# Alcalinity of ash                  11.153
# Magnesium                         203.989
# Total phenols                       0.392
# Flavanoids                          0.998
# Nonflavanoid phenols                0.015
# Proanthocyanins                     0.328
# Color intensity                     5.374
# Hue                                 0.052
# OD280/OD315 of diluted wines        0.504
# Proline                         99166.717
# dtype: float64

# The Proline column has an extremely high variance.
----------
# Print out the variance of the Proline column
print(wine.Proline.var())

# Apply the log normalization function to the Proline column
wine['Proline_log'] = np.log(wine.Proline)

# Check the variance of the normalized Proline column
print(wine.Proline_log.var())

# The np.log() function is an easy way to log normalize a column.
-----------
# Understanding your data is a crucial first step before deciding on the most appropriate standardization technique
-----------
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Create the scaler
scaler = StandardScaler()

# Subset the DataFrame you want to scale 
wine_subset = wine[['Ash','Alcalinity of ash', 'Magnesium']]

# Apply the scaler to wine_subset
wine_subset_scaled = scaler.fit_transform(wine_subset)

------------
# Split the dataset and labels into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Fit the k-nearest neighbors model to the training data
knn.fit(X_train, y_train)

# Score the model on the test data
print(knn.score(X_test, y_test))

# This accuracy definitely isn't poor, but let's see if we can improve it by standardizing the data
----------
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Instantiate a StandardScaler
scaler = StandardScaler()

# Scale the training and test features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the k-nearest neighbors model to the training data
knn.fit(X_train_scaled, y_train)

# Score the model on the test data
print(knn.score(X_test_scaled, y_test))
