import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# read csv
df = pd.read_csv('res/Mall_Customers.csv')
df.drop(["CustomerID"], axis=1, inplace=True)
# K-Mean algorithm
km = KMeans(n_clusters=5, max_iter=3000)
clusters = km.fit_predict(df.iloc[:, 1:])
df['cluster'] = clusters
clusters = df['cluster'].unique()
# init plt
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(projection='3d')
colors = ['red', 'green', 'blue', 'purple', 'magenta',
          'orange', 'brown', 'black', 'teal', 'cyan']
# draw dots with its cluster(color)
for cluster in clusters:
    ax.scatter(
        df['Annual Income (k$)'][df['cluster'] == cluster],
        df['Age'][df['cluster'] == cluster],
        df['Spending Score (1-100)'][df['cluster'] == cluster],
        s=50, c=colors[cluster])
# axis mods
plt.xlabel('Annual Income (k$)')
plt.ylabel('Age (years)')
ax.set_zlabel('Spending Score (1-100)')
ax.invert_xaxis()
plt.show()

#    CustomerID  Gender  Age  Annual Income (k$)  Spending (1-100)
# 0           1    Male   19                  15                39
# 1           2    Male   21                  15                81
# 2           3  Female   20                  16                 6
# 3           4  Female   23                  16                77
# 4           5  Female   31                  17                40
