import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 读取Excel文件的第二个表格（索引为1）
df = pd.read_excel('CowenKeltnerEmotionalVideos.xlsx', sheet_name=1)

# 提取标签列
labels_video = df[df.columns[0]]

# 删除标签列
df = df.drop(df.columns[0], axis=1)

# 提取标签并转换为整数
labels, _ = pd.factorize(df.idxmax(axis=1))

# # 使用PCA进行降维
# pca = PCA(n_components=3)
# df_pca = pca.fit_transform(df)

# 使用t-SNE进行降维
tsne = TSNE(n_components=3)
df_tsne = tsne.fit_transform(df)

# 创建一个新的DataFrame来保存降维后的数据和标签
df_new = pd.DataFrame(df_tsne, columns=['Dimension 1', 'Dimension 2', 'Dimension 3'])
df_new['Video'] = labels_video
df_new['Label'] = labels

# 保存DataFrame为CSV文件
df_new.to_csv('tsne_data.csv', index=False)

# 创建一个3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# # 绘制数据点，使用标签为数据点着色
# scatter = ax.scatter(df_pca[:, 0], df_pca[:, 1], df_pca[:, 2], c=labels)

# 绘制数据点，使用标签为数据点着色
scatter = ax.scatter(df_tsne[:, 0], df_tsne[:, 1], df_tsne[:, 2], c=labels)

# 添加颜色条
plt.colorbar(scatter)

# 显示图形
plt.show()