# 1. Mengimpor Library yang Dibutuhkan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Memuat Dataset
# Gantilah 'your_dataset.csv' dengan path ke dataset yang Anda gunakan (misalnya: di Google Drive atau di-upload)
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'  # Contoh dataset Titanic
df = pd.read_csv(url)

# 3. Menampilkan 5 baris pertama dari dataset
df.head()

# 4. Info umum tentang dataset (jumlah baris, kolom, tipe data)
df.info()

# 5. Statistik deskriptif untuk kolom numerik
df.describe()

# 6. Mengecek nilai yang hilang
df.isnull().sum()

# 7. Visualisasi distribusi data untuk kolom numerik
plt.figure(figsize=(12,8))
df.hist(bins=20, figsize=(12,10))
plt.show()

# 8. Korelasi antar fitur dengan heatmap
plt.figure(figsize=(12,8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

# 9. Visualisasi kategori dengan countplot (contoh: kolom 'Survived' yang kategorikal)
plt.figure(figsize=(8,6))
sns.countplot(x='Survived', data=df)
plt.show()

# 10. Boxplot untuk melihat distribusi kolom numerik berdasarkan kategori
plt.figure(figsize=(8,6))
sns.boxplot(x='Survived', y='Age', data=df)
plt.show()

# 11. Mengecek dan menghapus duplikasi data (jika ada)
print(f"Jumlah data duplikat: {df.duplicated().sum()}")
df = df.drop_duplicates()

# 12. Mengisi nilai yang hilang dengan metode tertentu (contoh: mengisi dengan median untuk 'Age')
df['Age'] = df['Age'].fillna(df['Age'].median())

# 13. Menampilkan beberapa baris terakhir setelah pengolahan
df.tail()
