import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 读取Excel文件的第二个表格（索引为1）
df = pd.read_excel('CowenKeltnerEmotionalVideos.xlsx', sheet_name=1)

# 提取标签列
labels_video = df[df.columns[0]]

# 删除标签列
df = df.drop(df.columns[0], axis=1)

# 提取标签并转换为整数
labels, emotion = pd.factorize(df.idxmax(axis=1))

# # 使用PCA进行降维
# pca = PCA(n_components=3)
# df_pca = pca.fit_transform(df)

# 使用t-SNE进行降维
tsne = TSNE(n_components=3)
df_tsne = tsne.fit_transform(df)

# 创建归一化器并将数据缩放到[-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
df_tsne_scaled = scaler.fit_transform(df_tsne)

# 创建一个新的DataFrame来保存降维后的数据和标签
df_new = pd.DataFrame(df_tsne_scaled, columns=['Dimension 1', 'Dimension 2', 'Dimension 3'])
df_new['Video'] = labels_video
df_new['Label'] = labels

# # 保存DataFrame为CSV文件
# df_new.to_csv('tsne_data.csv', index=False)

# 根据标签将数据分组，并计算每组的中心
df_new = df_new.drop('Video', axis=1)
centers = df_new.groupby('Label', as_index=False).mean()

# 计算每组的大小（即每组的点的数量）
sizes = df_new.groupby('Label').size()
centers['Size'] = sizes.values
centers['Emotion'] = emotion

centers.to_csv('tsne_centers.csv', index=False)

# 创建一个3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# # 绘制数据点，使用标签为数据点着色
# scatter = ax.scatter(df_pca[:, 0], df_pca[:, 1], df_pca[:, 2], c=labels)

# 绘制数据点，使用标签为数据点着色
scatter = ax.scatter(df_tsne_scaled[:, 0], df_tsne_scaled[:, 1], df_tsne_scaled[:, 2], c=labels, s=1)

# 绘制每个中心点，大小与该组的大小成比例
scatter_centers = ax.scatter(centers['Dimension 1'], centers['Dimension 2'], centers['Dimension 3'], c=centers['Label'], s=sizes*10, alpha=0.5)

# 添加颜色条
plt.colorbar(scatter)

# 显示图形
plt.show()