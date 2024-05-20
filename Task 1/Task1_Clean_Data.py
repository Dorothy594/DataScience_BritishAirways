import pandas as pd

# Load the data from a CSV file
df = pd.read_csv('BA_reviews.csv')

# Define the list of texts to remove
unnecessary_texts = ["✅ Trip Verified | ", "Not Verified | "]


# Function to remove all unnecessary texts from a string
def remove_unnecessary_texts(text, texts_to_remove):
    for unnecessary_text in texts_to_remove:
        text = text.replace(unnecessary_text, '')
    return text


# Apply the removal to each relevant column
df = df.applymap(lambda x: remove_unnecessary_texts(x, unnecessary_texts) if isinstance(x, str) else x)

# Save the cleaned data back to a CSV file
df.to_csv('cleaned_data.csv', index=False)

# Uncomment to execute in the Python Code Interpreter
# print(df.head())
