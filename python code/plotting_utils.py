import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# plot1 : function that plot two graphs, one of the predicted values and the other is for the original values
def plot1(df_original, df, idx, key):
    # Plot predicted values
    plt.subplot(1, 2, 1)
    plt.scatter(idx, df.loc[idx, key], color='red', marker='.')
    plt.title('Predicted Values')
    plt.xlabel('Index')
    plt.ylabel(key)

    # Plot original values
    plt.subplot(1, 2, 2)
    plt.scatter(idx, df_original.loc[idx, key], color='blue', marker='.')
    plt.title('Original Values')
    plt.xlabel('Index')
    plt.ylabel(key)

    # Adjust plot spacing
    plt.subplots_adjust(wspace=1)

    # Show plot
    plt.show()

# plot2 : function that plot one graph of the mean value and the actual values
def plot2(mean_value, idx, df, key):
    # Plot mean value in green
    plt.axhline(y=mean_value, color='green', linestyle='--', label='Mean')

    # Plot actual values in blue
    plt.scatter(idx, df.loc[idx, key], color='blue', label='Actual', marker='.')

    # Add legend and labels
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel(key)

    # Show plot
    plt.show()

# plot correlation
def corr_plot(key, value, col):
    df1 = pd.read_csv(key).head(value)
    df = pd.read_csv(key).head(value)
    # Compute the correlation matrix
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

    # find the correlation of each column with "other_sales"
    corr_matrix = df1.corr()[col]

    # print the correlation values
    print(corr_matrix)


# org_distribution_graph function to plot a graph for the original distribution for the given dataset and column
def org_distribution_graph(df, col):
    # Get value counts for Outlet_Location_Type column
    location_counts = df[col].value_counts()

    # Plot bar chart of value counts
    plt.bar(location_counts.index, location_counts.values)

    # Set plot title and axis labels
    plt.title(f'Real Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')

    # Show plot
    plt.show()

# after_distribution_graph function to plot a graph for the distribution for the given dataset and column after filling values
def after_distribution_graph(df, col):
    # Get value counts for Outlet_Location_Type column
    location_counts = df[col].value_counts()

    # Plot bar chart of value counts
    plt.bar(location_counts.index, location_counts.values)

    # Set plot title and axis labels
    plt.title(f'Distribution of {col} after Random forest method')
    plt.xlabel(col)
    plt.ylabel('Count')

    # Show plot
    plt.show()

