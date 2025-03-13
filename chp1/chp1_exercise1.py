# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:32:34 2024

@author: lich5
"""

#%% 综合练习
# 创建一个10x10的随机矩阵，归一化其所有元素到0到1之间。
import numpy as np
random_matrix = np.random.rand(10, 10)
normalized_matrix = (random_matrix - np.min(random_matrix)) / (np.max(random_matrix) - np.min(random_matrix))
print("归一化后的矩阵:")
print(normalized_matrix)

# 创建一个5x5的随机矩阵，找到其最大值的索引。
import numpy as np
try:
    matrix = np.random.rand(5, 5)
    print("随机矩阵:")
    print(matrix)
    flat_max_index = np.argmax(matrix)
    max_index = np.unravel_index(flat_max_index, matrix.shape)
    print("最大值的索引:", max_index)
    max_value = matrix[max_index]
    print("最大值:", max_value)
except Exception as e:
print(f"程序运行过程中出现错误: {e}")

# 创建一个10x3的随机矩阵，找出每行的最大值及其索引。
import numpy as np
matrix = np.random.rand(10, 3)
max_values = np.max(matrix, axis=1)
max_indices = np.argmax(matrix, axis=1)
for i in range(10):
    print(f"第 {i + 1} 行的最大值是 {max_values[i]}，索引是 {max_indices[i]}")
print("生成的随机矩阵为：")
print(matrix)

# 创建一个长度为20的随机数组，找出其第二大的元素。
import random
def find_second_largest(arr):
    unique_arr = list(set(arr))
    if len(unique_arr) < 2:
        return None
    unique_arr.sort(reverse=True)
    return unique_arr[1]
random_arr = [random.randint(1, 100) for _ in range(20)]
second_largest = find_second_largest(random_arr)
print("随机数组:", random_arr)
if second_largest is not None:
    print("第二大的元素:", second_largest)
else:
print("数组中没有第二大的元素。")

# 创建一个3x3的随机矩阵，将其转换为仅包含0和1的矩阵（根据某个自定义阈值）。
import numpy as np
threshold = 0.5
random_matrix = np.random.rand(3, 3)
binary_matrix = (random_matrix > threshold).astype(int)
print("随机矩阵:")
print(random_matrix)
print("转换后的 0 - 1 矩阵:")
print(binary_matrix)

# 创建一个包含1000个元素的数组，将其中的偶数替换为-1。
arr = list(range(1000))
arr = [-1 if i % 2 == 0 else i for i in arr]
print(arr)

# 创建一个5x5的随机矩阵，并将其中的奇数行逆序排列。
import numpy as np
matrix = np.random.randint(0, 100, (5, 5))
print("原始矩阵:")
print(matrix)
matrix[1::2] = matrix[1::2, ::-1]
print("\n奇数行逆序后的矩阵:")
print(matrix)

# 创建一个长度为10的数组，查找数组中连续大于0.5的元素段。
import random
array = [random.random() for _ in range(10)]
print("生成的数组:", array)
segments = []
current_segment = []
for num in array:
    if num > 0.5:
        current_segment.append(num)
    else:
        if current_segment:
            segments.append(current_segment)
            current_segment = []
if current_segment:
    segments.append(current_segment)
print("连续大于 0.5 的元素段:")
for segment in segments:
print(segment)

# 使用numpy计算两个随机数组之间的欧氏距离。
import numpy as np
array1 = np.random.rand(10)
array2 = np.random.rand(10)
euclidean_distance = np.linalg.norm(array1 - array2)
print("两个随机数组的欧氏距离为:", euclidean_distance)

# 生成一个10x10的随机矩阵，查找其局部最大值（即比周围八个元素都大的值）。
import numpy as np
matrix = np.random.rand(10, 10)
local_maxima = []
for i in range(1, 9):
    for j in range(1, 9):
        current = matrix[i, j]
        neighbors = [
            matrix[i - 1, j - 1], matrix[i - 1, j], matrix[i - 1, j + 1],
            matrix[i, j - 1], matrix[i, j + 1],
            matrix[i + 1, j - 1], matrix[i + 1, j], matrix[i + 1, j + 1]
        ]
        if all(current > neighbor for neighbor in neighbors):
            local_maxima.append(current)
print("局部最大值:", local_maxima)

