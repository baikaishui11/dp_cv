'''
1.你有一个凸的 n 边形，其每个顶点都有一个整数值。给定一个整数数组 values ，其中 values[i] 是第 i 个顶点的值（即 顺时针顺序 ）。
假设将多边形 剖分 为 n - 2 个三角形。对于每个三角形，该三角形的值是顶点标记的乘积，三角剖分的分数是进行三角剖分后所有 n - 2 个三角形的值之和。
返回 多边形进行三角剖分后可以得到的最低分 。
'''


def minScoreTriangulation(values):
    n = len(values)
    # 初始化DP数组，dp[i][j]表示顶点i到j的子多边形的最低剖分分数
    dp = [[0] * n for _ in range(n)]

    # 按子多边形的长度（j-i）从小到大计算，长度至少为2（3个顶点）
    for length in range(2, n):  # length = j - i，范围2~n-1（对应3~n个顶点）
        for i in range(n - length):  # i的最大值为n-1 - length（确保j = i+length < n）
            j = i + length
            # 初始化为无穷大，后续取最小值
            dp[i][j] = float('inf')
            # 枚举i和j之间的所有k，计算最小分数
            for k in range(i + 1, j):
                current = dp[i][k] + dp[k][j] + values[i] * values[k] * values[j]
                if current < dp[i][j]:
                    dp[i][j] = current
    # 整个n边形的顶点范围是0~n-1，返回dp[0][n-1]
    return dp[0][n - 1]


if __name__ == "__main__":
    lst = [4, 5, 6, 7, 8, 2]
    print(minScoreTriangulation(lst))