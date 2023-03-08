import matplotlib as plt

# string1 = 'Other_Sales'
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