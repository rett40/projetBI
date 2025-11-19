import pandas as pd
p = '.venv\\app\\data\\dataset.csv'
try:
    df = pd.read_csv(p, encoding='utf-8')
except Exception as e:
    print('Could not read', p, e)
    raise
print('Total rows:', len(df))
print('Unique URLs:', df['url'].nunique())
print('\nStatus codes:')
print(df['status_code'].value_counts(dropna=False).to_string())
print('\nTop errors:')
print(df['error'].value_counts().head(20).to_string())
