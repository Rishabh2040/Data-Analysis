import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file = pd.read_csv("32_Constituency_Data_Summary_Report.csv")

# Basic Info
print("\n▶ First 5 Rows:")
print(file.head())

print("\n▶ Last 5 Rows:")
print(file.tail())

print("\n▶ Shape of Data:")
print(file.shape)

print("\n▶ Column Names:")
print(file.columns)

print("\n▶ Dataset Info:")
print(file.info())

print("\n▶ Statistical Summary:")
print(file.describe())

# Mean
print("\n▶ Mean of Numerical Columns:")
print(file[['Men', 'Women', 'Third Gender', 'Total']].mean())

# Median
print("\n▶ Median of Numerical Columns:")
print(file[['Men', 'Women', 'Third Gender', 'Total']].median())

# Mode
print("\n▶ Mode of Numerical Columns:")
print(file[['Men', 'Women', 'Third Gender', 'Total']].mode().iloc[0])  # Using .iloc[0] to get first mode row

# Missing Values
print("\n▶ Missing Values:")
print(file.isnull().sum())

# Drop missing values
file = file.dropna()
print("\n▶ Missing Values After Drop:")
print(file.isnull().sum())

# Drop duplicates
file = file.drop_duplicates()
print("\n▶ Data After Dropping Duplicates:")
print(file.head())

# Plot: Count of Categories
plt.figure(figsize=(10, 6))
sns.countplot(data=file, x='Category', palette='Set2')
plt.title("Count of Records by Category", fontsize=14)
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot: Gender distribution by category
file_gender = file.groupby('Category')[['Men', 'Women', 'Third Gender']].sum().reset_index()
file_gender.plot(x='Category', kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
plt.title("Gender Distribution per Category")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Heatmap of Correlation
plt.figure(figsize=(8, 6))
corr = file[['Men', 'Women', 'Third Gender', 'Total']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Pie Chart: Gender-wise Distribution of Voters
gender_totals = file[['Men', 'Women', 'Third Gender']].sum()

plt.figure(figsize=(8, 8))
plt.pie(gender_totals, labels=gender_totals.index, autopct='%1.1f%%', startangle=140,
        colors=['skyblue', 'lightcoral', 'lightgreen'])
plt.title("Gender-wise Distribution of Voters")
plt.axis('equal')
plt.tight_layout()
plt.show()

# Pie Chart: Proportion of Records by Category
category_counts = file['Category'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140,
        colors=sns.color_palette('Set2'))
plt.title("Proportion of Records by Category")
plt.axis('equal')
plt.tight_layout()
plt.show()
