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


def t1():
    lst = [4, 5, 6, 7, 8, 2]
    print(minScoreTriangulation(lst))


'''
2.给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
请你将两个数相加，并以相同形式返回一个表示和的链表。
你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
'''


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        # 创建一个哑节点作为结果链表的起始点
        dummy = ListNode(0)
        current = dummy
        carry = 0  # 进位

        # 遍历两个链表，直到都为空且没有进位
        while l1 or l2 or carry:
            # 获取当前节点的值，若节点为空则取0
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0

            # 计算当前位的总和（包括进位）
            total = val1 + val2 + carry
            # 计算当前位的值和新的进位
            current_val = total % 10
            carry = total // 10

            # 创建新节点并添加到结果链表
            current.next = ListNode(current_val)
            current = current.next

            # 移动到下一个节点
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next

        # 返回结果链表的头节点（哑节点的下一个节点）
        return dummy.next


def create_linked_list(nums):
    dummy = ListNode(0)
    current = dummy
    for num in nums:
        current.next = ListNode(num)
        current = current.next
    return dummy.next


# 辅助函数：将链表转为列表（便于验证结果）
def linked_list_to_list(head):
    result = []
    current = head
    while current:
        result.append(current.val)
        current = current.next
    return result


# 测试用例执行函数
def test_case(l1_nums, l2_nums, expected_nums):
    l1 = create_linked_list(l1_nums)
    l2 = create_linked_list(l2_nums)
    solution = Solution()
    result_head = solution.addTwoNumbers(l1, l2)
    result_nums = linked_list_to_list(result_head)
    print(f"输入链表1: {l1_nums}（逆序表示 {int(''.join(map(str, reversed(l1_nums)))) if l1_nums else 0}）")
    print(f"输入链表2: {l2_nums}（逆序表示 {int(''.join(map(str, reversed(l2_nums)))) if l2_nums else 0}）")
    print(f"预期结果: {expected_nums}（逆序表示 {int(''.join(map(str, reversed(expected_nums)))) if expected_nums else 0}）")
    print(f"实际结果: {result_nums}")
    print("测试通过" if result_nums == expected_nums else "测试失败")
    print("-" * 50)


def t2():
    test_case([2, 4, 3], [5, 6, 4], [7, 0, 8])
    test_case([9, 9, 9, 9, 9, 9, 9], [9, 9, 9, 9], [8, 9, 9, 9, 0, 0, 0, 1])


if __name__ == "__main__":
    # t1()
    t2()
