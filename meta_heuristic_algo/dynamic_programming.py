from lib2to3.pgen2.token import NUMBER
import numpy as np

KNAPSACK_CAP = 750
NUMBER_OF_OBJECTS = 15
KNAPSACK_WEIGHT = [70, 73, 77, 80, 82, 87, 90, 94, 98, 106, 110, 113, 115, 118, 120]
KNAPSACK_PROFIT = [135, 139, 149, 150, 156, 163, 173, 184, 192, 201, 210, 214, 221, 229, 240]

OPTIMAL_SELECTION =  [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]
OPTIMAL_PROFIT = 1458

# https://leetcode.com/problems/target-sum/discuss/1157060/01-knapsack
dp = np.zeros((NUMBER_OF_OBJECTS + 1, KNAPSACK_CAP + 1), dtype=int)
for i in range(1, NUMBER_OF_OBJECTS + 1, 1):
    iknap = i - 1
    for j in range(KNAPSACK_CAP + 1):
        if  j < KNAPSACK_WEIGHT[iknap]:
            dp[i][j] = dp[iknap][j]
            ()
        else:
            dp[i][j] = max(dp[iknap][j], KNAPSACK_PROFIT[iknap] + dp[iknap][(j - KNAPSACK_WEIGHT[iknap])])
            ()
        ()
    ()

iscorrect = dp[NUMBER_OF_OBJECTS, KNAPSACK_CAP] == OPTIMAL_PROFIT
print(dp[NUMBER_OF_OBJECTS, KNAPSACK_CAP])
if iscorrect:
    print("Is answer correct? YES")
else:
    print("Is answer correct? NO")