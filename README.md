# Exploratory Data Analysis of the Iris Dataset

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project provides a comprehensive exploratory data analysis (EDA) of the classic Iris dataset. The analysis covers data inspection, calculation of univariate summary statistics, correlation analysis, and the creation of various visualizations to understand the dataset's characteristics and the relationships between its features.

---

## 📊 Key Analysis & Features

* **Summary Statistics:** Calculated key statistical measures (mean, median, quartiles, standard deviation, etc.) for the 'sepal width' attribute, grouped by each species. The results are saved in `dist/statistics.csv`.
* **Correlation Analysis:** Computed a Pearson correlation matrix for the four quantitative features to identify the strength and direction of their linear relationships. The matrix is saved in `dist/correlations.csv`.
* **Data Visualization:** Created a variety of plots to visualize the data, including:
    * Bar plots to show the class distribution.
    * Histograms to understand the distribution of individual features.
    * Box plots to compare feature distributions across different species.
    * 2D and 3D scatter plots to visualize the separability of the classes based on feature pairs.
    * Probability Density Function (PDF) plots to model the distribution of features for each species.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare the Dataset:**
    * Create a folder named `data` in the project's root directory.
    * Download the Iris dataset (e.g., from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)) and place the data file inside the `data` folder.

4.  **Execute the Analysis:**
    Run the main script from the root directory. The script will read the dataset, perform all calculations, and generate the output files and plots.
    ```bash
    python src/run.py
    ```
    * The output CSV files (`statistics.csv`, `correlations.csv`) will be saved in the `dist` folder.
    * All visualizations will be displayed or saved to the `dist` folder.

---

## 📄 License

This project is licensed under the MIT License.
