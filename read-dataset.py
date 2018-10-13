import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib
import cv2 as cv


with open('dataset.csv', 'r') as csv_file:
    result = csv.reader(csv_file)
    rows = []
    
    # đọc từng dòng của file và thêm vào list rows, mỗi phần tử của list là một dòng
    for row in result:
        rows.append(row)

#print(np.shape(rows[100000]))
#letter = rows[30000]
x = np.array([int(j) for j in letter[0:]])
x = x.reshape(28, 28)

#print(letter)
plt.imshow(x)