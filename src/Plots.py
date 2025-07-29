#Name:Hesameddin Fathi
#Student Number:40330795

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("iris.csv")

#------------------------------------------------------------------------------------------
#Distribution of Labels
sns.countplot(x='variety', data=df)
plt.title('distribution of species labels')
plt.xlabel('species')
plt.ylabel('num')
plt.show()

#------------------------------------------------------------------------------------------
#Histograms
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(df['petal.length'], bins=20, color='blue')
plt.xlabel('petal length')
plt.ylabel('num')
plt.title('petal length-histogram')
#sepal width
plt.subplot(1, 2, 2)
plt.hist(df['sepal.width'], bins=20, color='red')
plt.xlabel('sepal width')
plt.ylabel('num')
plt.title('sepal width-histogram')
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------
#Histograms 3D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
df = pd.read_csv("iris.csv")
x = df['petal.length']
y = df['sepal.width']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#3D
hist, xedges, yedges = np.histogram2d(x, y, bins=20)
xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)
dx = xedges[1] - xedges[0]
dy = yedges[1] - yedges[0]
dz = hist.flatten()
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', alpha=0.5)
ax.set_xlabel('petal length')
ax.set_ylabel('sepal width')
ax.set_zlabel('num')
plt.title('3D histogram')

plt.show()

#------------------------------------------------------------------------------------------
#Box Plots part 1
sns.boxplot(x='variety', y='petal.length', data=df)
plt.title('distribution of petal length for each species')

plt.figure()
sns.boxplot(x='variety', y='sepal.width', data=df)
plt.title('distribution of sepal width for each species')

plt.show()

#------------------------------------------------------------------------------------------
#Box Plots part 2
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
from sklearn.datasets import load_iris

def boxplot_2d(x, y, ax, whis=1.5):
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]

    # The box
    box = Rectangle(
        (xlimits[0], ylimits[0]),
        (xlimits[2] - xlimits[0]),
        (ylimits[2] - ylimits[0]),
        ec='k',
        zorder=0
    )
    ax.add_patch(box)

    # The x median
    vline = Line2D(
        [xlimits[1], xlimits[1]], [ylimits[0], ylimits[2]],
        color='k',
        zorder=1
    )
    ax.add_line(vline)

    # The y median
    hline = Line2D(
        [xlimits[0], xlimits[2]], [ylimits[1], ylimits[1]],
        color='k',
        zorder=1
    )
    ax.add_line(hline)

    # The central point
    ax.plot([xlimits[1]], [ylimits[1]], color='k', marker='o')

    # The x-whisker
    iqr = xlimits[2] - xlimits[0]

    # Left
    left = np.min(x[x > xlimits[0] - whis * iqr])
    whisker_line = Line2D(
        [left, xlimits[0]], [ylimits[1], ylimits[1]],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [left, left], [ylimits[0], ylimits[2]],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_bar)

    # Right
    right = np.max(x[x < xlimits[2] + whis * iqr])
    whisker_line = Line2D(
        [right, xlimits[2]], [ylimits[1], ylimits[1]],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [right, right], [ylimits[0], ylimits[2]],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_bar)

    # The y-whisker
    iqr = ylimits[2] - ylimits[0]

    # Bottom
    bottom = np.min(y[y > ylimits[0] - whis * iqr])
    whisker_line = Line2D(
        [xlimits[1], xlimits[1]], [bottom, ylimits[0]],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0], xlimits[2]], [bottom, bottom],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_bar)

    # Top
    top = np.max(y[y < ylimits[2] + whis * iqr])
    whisker_line = Line2D(
        [xlimits[1], xlimits[1]], [top, ylimits[2]],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0], xlimits[2]], [top, top],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_bar)

    # Outliers
    mask = (x < left) | (x > right) | (y < bottom) | (y > top)
    ax.scatter(
        x[mask], y[mask],
        facecolors='none', edgecolors='k'
    )

# Load the Iris dataset
iris = load_iris()
petal_length = iris.data[:, 2]
sepal_width = iris.data[:, 1]

# Create figure and axes
fig, ax = plt.subplots(figsize=(6, 6))

# Creating the box plot
boxplot_2d(petal_length, sepal_width, ax=ax, whis=1.5)
ax.set_xlabel('Petal Length')
ax.set_ylabel('Sepal Width')
ax.set_title('2D Box Plot')

plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------
#Quantile Plot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
petal_length = iris.data[:, 2]
sepal_width = iris.data[:, 1]

def quantile_plot(data, feature_name):
    # Sort the data
    sorted_data = np.sort(data)
    N = len(sorted_data)
    
    # Calculate fi
    fi = (np.arange(1, N + 1) - 0.5) / N
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fi,sorted_data, marker='o', linestyle='-', color='b')
    plt.xlabel('fi')
    plt.ylabel(feature_name)
    plt.title(f'Quantile Plot of {feature_name}')
    plt.grid(True)
    plt.show()

# Generate quantile plots for petal length and sepal width
quantile_plot(petal_length, 'Petal Length')
quantile_plot(sepal_width, 'Sepal Width')

#------------------------------------------------------------------------------------------
#Scatter Plots part 1
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
# Map target numbers to species names
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
iris_df['species'] = iris_df['species'].map(species_map)
# Define feature pairs for scatter plots
feature_pairs = [
    ('sepal length (cm)', 'sepal width (cm)'),
    ('sepal length (cm)', 'petal length (cm)'),
    ('sepal length (cm)', 'petal width (cm)'),
    ('sepal width (cm)', 'petal length (cm)'),
    ('sepal width (cm)', 'petal width (cm)'),
    ('petal length (cm)', 'petal width (cm)')
]
# Plotting
plt.figure(figsize=(15, 10))
for i, (x_feature, y_feature) in enumerate(feature_pairs, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(data=iris_df, x=x_feature, y=y_feature, hue='species', palette='deep')
    plt.title(f'{x_feature} vs {y_feature}')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)

plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------
#Scatter Plots 3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

# Map target numbers to species names
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
iris_df['species'] = iris_df['species'].map(species_map)
# 3D Scatter Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# Define colors for each species
colors = {'setosa': 'r', 'versicolor': 'g', 'virginica': 'b'}
# Plot each species with different colors
for species, color in colors.items():
    subset = iris_df[iris_df['species'] == species]
    ax.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'], subset['petal length (cm)'],
               c=color, label=species, s=50)
# Labels and title
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Sepal Width (cm)')
ax.set_zlabel('Petal Length (cm)')
ax.set_title('3D Scatter Plot of Iris Dataset')
ax.view_init(elev=20, azim=135)  # Adjust the view angle for better visualization
ax.legend()
plt.show()

#---------------------------------------------------------------------------------------------
#Probability Distributions
# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

# Map target numbers to species names
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
iris_df['species'] = iris_df['species'].map(species_map)

# Plotting the PDF for petal length for each species
plt.figure(figsize=(10, 6))
sns.kdeplot(data=iris_df, x='petal length (cm)', hue='species', common_norm=False, fill=True)

# Labels and title
plt.xlabel('Petal Length (cm)')
plt.ylabel('Density')
plt.title('Probability Density Function of Petal Length by Species')
plt.legend(title='Species')

plt.show()





