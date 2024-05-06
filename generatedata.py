import csv
import random
import numpy as np

# Define the centers
centers = [(240, 135), (720, 135), (240, 405), (720, 405)]

# Generate the points
points = []
for center in centers:
    for _ in range(250):  # Generate 250 points around each center
        x = int(np.random.normal(center[0], 60))  # 60 is the standard deviation
        y = int(np.random.normal(center[1], 60))
        points.append((x, y))

# Write the points to a CSV file
with open('points.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["x", "y"])  # Write the column names
    writer.writerows(points)