import pandas as pd

# Load the csv file
df = pd.read_csv('train_liar.csv')

# Define a function to map the categories
def map_category(category):
    if category in ['pants-fire', 'mostly-false','false','FALSE']:
        return 'fake'
    elif category in ['half-true', 'mostly-true','barely-true','true','TRUE']:
        return 'reliable'
    else:
        return category

# Apply the function to the 'category' column
df['category'] = df['category'].apply(map_category)

# Save the dataframe back to the csv file
df.to_csv('train_liar_update.csv', index=False)
