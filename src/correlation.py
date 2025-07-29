import os
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Compute the correlation matrix
correlation_matrix = data.corr(method='pearson')

# Define the relative path to the 'dist' folder
dist_path = os.path.join('..', 'dist', 'correlations.csv')

# Save the correlation matrix to a CSV file in the 'dist' folder
correlation_matrix.to_csv(dist_path, header=False, index=False)