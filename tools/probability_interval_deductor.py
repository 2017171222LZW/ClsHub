import math
import random
# 适用于莫氏硬度模拟的随机数生成器
# 基本原理：根据给定数值区间，以及区间的目标累计概率，生成一维高斯分布，从生成的高斯分布中采样，生成随机数
def normal_distribution_parameters_random_sample(a, b, area, precision=0.0001, max_iter=100):  
    # 定义计算累积概率的函数
    def cdf(x, mu, sigma):
        # x表示输入值，mu表示正态分布的均值，sigma表示正态分布的标准差
        return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))
      
    # 使用二分法计算满足区间内面积为area的正态分布参数
    mu = (a + b) / 2  # 正态分布的均值
    sigma = (b - a) / 4  # 正态分布的标准差
    left = mu - sigma  # 初始化左边界为均值减去标准差
    right = mu + sigma  # 初始化右边界为均值加上标准差
    mid = (left + right) / 2 # 初始化中点为左边界和右边界的平均值
    iter_count = 0
    # 判断累积概率的差值是否大于给定精度，并且迭代次数是否小于最大迭代次数
    while abs(cdf(mid, mu, sigma) - area) > precision and iter_count < max_iter:
        # 如果当前累积概率小于所需的累积概率面积
        if cdf(mid, mu, sigma) < area:
            # 更新左边界为中点
            left = mid
        else:
            # 更新右边界为中点
            right = mid
        # 更新中点为新的左边界和右边界的平均值
        mid = (left + right) / 2
        iter_count += 1
    return random.gauss(mu, sigma)


# 测试用例
a = 3
b = 7
area = 0.8
# 随机生成20个数
for i in range(20):
    random_number = normal_distribution_parameters_random_sample(a, b, area)  
    print(abs(random_number))  
