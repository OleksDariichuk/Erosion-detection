def count_islands(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    def dfs(i, j):
        if i < 0 or j < 0 or i >= rows or j >= cols or matrix[i][j] == 0:
            return
        matrix[i][j] = 0
        dfs(i+1, j)
        dfs(i-1, j)
        dfs(i, j+1)
        dfs(i, j-1)

    count = 0
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1:
                dfs(i, j)
                count += 1
    return count

# matrix = [[0, 1, 0], [0, 0, 0], [0, 1, 1]]
# print(count_islands(matrix))  

# matrix = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]
# print(count_islands(matrix))  

# matrix = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1]]
# print(count_islands(matrix))  
