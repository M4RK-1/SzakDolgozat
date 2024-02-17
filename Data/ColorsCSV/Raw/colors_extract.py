import pandas as pd

# Attempt to load the newly uploaded CSV file again, this time skipping bad lines
df_new_updated = pd.read_csv('clean_colors.csv', delimiter=',')

# Save the first column to a new CSV file
df_new_updated.iloc[:, 0].to_csv('features.csv', index=False)

# Save the second column to a new CSV file
df_new_updated.iloc[:, 1].to_csv('all_labels.csv', index=False)

# Save the first 100 entries of the second column to a different CSV file
df_new_updated.iloc[:100, 1].to_csv('cut_labels.csv', index=False)

# Save all the unique options from the second column to a different CSV file
df_new_updated.iloc[:, 1].drop_duplicates().to_csv('unique_labels.csv', index=False)
