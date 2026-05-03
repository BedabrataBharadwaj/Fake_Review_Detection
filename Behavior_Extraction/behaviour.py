import pandas as pd
import numpy as np
import re
import emoji

df = pd.read_csv("edited_raw.csv")

# Clean column names
df.columns = df.columns.str.strip()
print("Original columns:", df.columns.tolist())

# Rename columns properly if needed
df = df.rename(columns={
    'review_text label': 'review_text',   # if this is one merged column
    'label': 'label'
})

# If columns are still weird, inspect again
print("Updated columns:", df.columns.tolist())

# Convert date
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

# Review frequency
df['review_frequency'] = df.groupby('user_id')['review_text'].transform('count')

# Burstiness
df = df.sort_values(['user_id', 'date'])
df['time_diff'] = df.groupby('user_id')['date'].diff().dt.total_seconds()
df['burst_flag'] = df['time_diff'].apply(lambda x: 1 if pd.notnull(x) and x < 3600 else 0)
df['burstiness'] = df.groupby('user_id')['burst_flag'].transform('sum')
df.drop(columns=['time_diff', 'burst_flag'], inplace=True)

# Rating deviation
product_avg = df.groupby('product_id')['rating'].transform('mean')
df['rating_deviation'] = abs(df['rating'] - product_avg)

# Punctuation intensity
def punctuation_intensity(text):
    text = str(text)
    punct = len(re.findall(r'[!?]', text))
    length = len(text)
    return punct / length if length > 0 else 0

df['punctuation_intensity'] = df['review_text'].apply(punctuation_intensity)

# Capitalization ratio
def capitalization_ratio(text):
    text = str(text)
    upper = sum(1 for c in text if c.isupper())
    total = len(text)
    return upper / total if total > 0 else 0

df['capitalization_ratio'] = df['review_text'].apply(capitalization_ratio)

# Save
df.to_csv("dataset_with_behavior_features.csv", index=False)

print("Done. File saved as dataset_with_behavior_features.csv")