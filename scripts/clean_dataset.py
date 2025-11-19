import pandas as pd
import os
p = '.venv\\app\\data\\dataset.csv'
df = pd.read_csv(p, encoding='utf-8')
# keep rows where url looks like a URL
mask = df['url'].astype(str).str.startswith(('http://','https://','www.'))
clean = df[mask].copy()
# drop duplicates by url
clean = clean.drop_duplicates(subset=['url'])
out = '.venv\\app\\data\\dataset_cleaned.csv'
clean.to_csv(out, index=False, encoding='utf-8')
print('Wrote', out)
print('Total rows original:', len(df))
print('Kept rows:', len(clean))
print('\nStatus code counts (cleaned):')
print(clean['status_code'].value_counts(dropna=False).to_string())
