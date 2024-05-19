import pandas as pd
import numpy as np

def calculate_distance(points1,points2):
    return np.sqrt(np.sum(np.square(points1-points2)))

# 读取CSV文件
data = pd.read_csv('tsne_centers.csv')
result = pd.DataFrame(columns=['Closest Emotion 1 Label', 'Closest Emotion 2 Label'])

unique_emotions = data['Emotion'].unique()

for emotion in unique_emotions:
    # 获取当前情绪的坐标
    current_emotion_data = data[data['Emotion'] == emotion]
    current_emotion_point = current_emotion_data[['Dimension 1', 'Dimension 2', 'Dimension 3']].values[0]
    
    # 计算当前情绪与其他所有情绪的距离
    distances = []
    for other_emotion in unique_emotions:
        if other_emotion != emotion:
            other_emotion_data = data[data['Emotion'] == other_emotion]
            other_emotion_point = other_emotion_data[['Dimension 1', 'Dimension 2', 'Dimension 3']].values[0]
            distance = calculate_distance(current_emotion_point, other_emotion_point)
            distances.append((other_emotion, distance))
    
    # 找出最近的两个情绪
    distances.sort(key=lambda x: x[1])
    closest_emotions = distances[:2]

    closest_emotion_1_label = data[data['Emotion'] == closest_emotions[0][0]]['Label'].values[0]
    closest_emotion_2_label = data[data['Emotion'] == closest_emotions[1][0]]['Label'].values[0]

    # 将结果添加到DataFrame中
    result = result._append({'Closest Emotion 1 Label': closest_emotion_1_label, 'Closest Emotion 2 Label': closest_emotion_2_label}, ignore_index=True)
    print(closest_emotion_1_label, closest_emotion_2_label)
    
    #print(f"The closest emotions to {emotion} are {closest_emotions[0][0]} and {closest_emotions[1][0]}")

# 将结果保存为CSV文件
result.to_csv('closest_emotions_labels.csv', index=False)

