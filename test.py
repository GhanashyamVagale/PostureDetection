# pip install matplotlib

import matplotlib.pyplot as plt
import pandas as pd  # If your data is in a DataFrame

# Example for loading data from a CSV file
df = pd.read_csv(
    '/Users/ghanashyamvagale/Desktop/Posture Detection/Dataset_spine.csv')

# Extract the two columns you want to plot
subset_df = df.iloc[:2]
x_values = subset_df['Col1']  # Replace 'Column1' with the actual column name
y_values = subset_df['Col2']  # Replace 'Column2' with the actual column name

# Create the scatter plot
plt.figure(figsize=(8, 6))  # Optional: Set the figure size
# plt.scatter(x_values, y_values, c='blue', marker='o', label='Data Points')  # You can customize the color, marker, and label
plt.plot(x_values, y_values, marker='o', linestyle='-')

# Add labels and title
plt.xlabel('std Neck angle')
plt.ylabel('std Torso angle')
plt.title('Scatter Plot of Std Neck angle vs. Torso angle')

# Add a legend (if needed)
# plt.legend()

# Show the plot
plt.show()
