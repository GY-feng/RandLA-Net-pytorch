import random

# 1. 生成 1 到 100 的数字列表
num_list = list(range(1, 234))
# 2. 随机打乱列表（原地打乱，效率高）
# random.shuffle(num_list)
# # 3. 循环输出，一行一个数字
for num in num_list:
    print(num)

    #  python train.py --dataset SlopeLAS --epochs 100 --batch_size 4 --adam_lr 0.01