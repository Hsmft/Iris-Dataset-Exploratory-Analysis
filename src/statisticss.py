import pandas as pd
import os
import numpy as np
df = pd.read_csv("iris.csv")
grouped_df = df.groupby('variety')

for name, group in grouped_df:
    missing_values = group['sepal.width'].isnull().sum()
    print(f"Missing {name}: {missing_values}")
for name, group in grouped_df:
    min_sepal_width = group['sepal.width'].min()
    print(f"Min {name}: {min_sepal_width}")
for name, group in grouped_df:
    q1 = group['sepal.width'].quantile(0.25)
    print(f"q1 {name}: {q1}")
for name, group in grouped_df:
    median_sepal_width = group['sepal.width'].median()
    print(f"Median {name}: {median_sepal_width}")
for name, group in grouped_df:
    q3 = group['sepal.width'].quantile(0.75)
    print(f"q3 {name}: {q3}")
for name, group in grouped_df:
    percentile_95 = group['sepal.width'].quantile(0.95)
    print(f"p95 {name}: {percentile_95}")
for name, group in grouped_df:
    max_sepal_width = group['sepal.width'].max()
    print(f"Max {name}: {max_sepal_width}")
for name, group in grouped_df:
    mean_sepal_width = group['sepal.width'].mean()
    print(f"Mean {name}: {mean_sepal_width}")
for name, group in grouped_df:
    data_range = group['sepal.width'].max() - group['sepal.width'].min()
    print(f"Range {name}: {data_range}")
for name, group in grouped_df:
    Q1 = group['sepal.width'].quantile(0.25)
    Q3 = group['sepal.width'].quantile(0.75)
    IQR = Q3 - Q1
    print(f"IQR {name}: {IQR}")
for name, group in grouped_df:
    std_dev_sample = group['sepal.width'].std()
    print(f"Std {name}: {std_dev_sample}")
for name, group in grouped_df:
    std_dev_population = group['sepal.width'].std(ddof=0)
    print(f"Std_pop{name}: {std_dev_population}")
for name, group in grouped_df:
    median = group['sepal.width'].median()
    absolute_deviations = np.abs(group['sepal.width'] - median)
    mad = np.median(absolute_deviations)
    print(f"MAD {name}: {mad}")

results = []
for name, group in grouped_df:
    missing_values = group['sepal.width'].isnull().sum()
    min_sepal_width = group['sepal.width'].min()
    q1 = group['sepal.width'].quantile(0.25)
    median_sepal_width = group['sepal.width'].median()
    q3 = group['sepal.width'].quantile(0.75)
    percentile_95 = group['sepal.width'].quantile(0.95)
    max_sepal_width = group['sepal.width'].max()
    mean_sepal_width = group['sepal.width'].mean()
    data_range = group['sepal.width'].max() - group['sepal.width'].min()
    IQR = q3 - q1
    std_dev_sample = group['sepal.width'].std()
    std_dev_population = group['sepal.width'].std(ddof=0)
    median = group['sepal.width'].median()
    absolute_deviations = np.abs(group['sepal.width'] - median)
    mad = np.median(absolute_deviations)

    results.append([name, missing_values, min_sepal_width, q1, median_sepal_width, q3, percentile_95, max_sepal_width, mean_sepal_width, data_range, IQR, std_dev_sample, std_dev_population, mad])

df_results = pd.DataFrame(results, columns=['label', 'missing', 'min', 'q1', 'med', 'q3', 'p95', 'max', 'mean', 'range', 'iqr', 'std', 'std_pop', 'mad'])
# Define the relative path to the 'dist' folder
dist_path = os.path.join('..', 'dist', 'statistics.csv')

# Save the correlation matrix to a CSV file in the 'dist' folder
df_results.to_csv(dist_path, header=False, index=False)

# Print the DataFrame in a tabular format with lines
print(df_results.to_string(index=False))

