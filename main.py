# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading USA Housing Dataset
data = pd.read_csv("USA_Housing.csv")
X = data.select_dtypes(exclude=["object"])

# Data visualization using seaborn

plt.figure(figsize=(15, 10))

for i, column in enumerate(X.columns[:-1], 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(data=X, x=column, y="Price")
    plt.title(f"Price vs. {column}")

plt.tight_layout()
plt.show()

# Compute and display Pearson Correlation Coefficients for each independent variable against the dependent
correlations = X.corr()['Price'].drop('Price')
sorted_correlations = correlations.abs().sort_values(ascending=False)  # Sorting vals in descending

print("print(f'Independent variables in the dataset sorted by the strength of their linear relationship with the",
      " \"Price\" column, irrespective of whether that relationship is positive or negative.\n{}')"
      .format(sorted_correlations))

# Visualizing correlations
plt.figure(figsize=(20, 15))

for i, column in enumerate(sorted_correlations.index, 1):
    plt.subplot(2, 3, i)
    sns.regplot(data=X,
                x=column,
                y="Price",
                line_kws={"color": "red",
                          "label": f"Pearson Correlation: {sorted_correlations[column]:.4f}"})

    plt.title(f"Price vs. {column}")
    plt.legend()
plt.tight_layout()
plt.show()
