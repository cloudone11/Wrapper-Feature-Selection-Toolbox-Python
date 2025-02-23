# import numpy as np

# # 假设 arr 是一个形状为 (136,) 的一维数组
# arr = np.arange(136)  # 示例数据

# # 将一维数组转换为形状为 (1, 136) 的二维数组
# arr_2d = arr.reshape(1, -1)  # -1 表示自动计算行数

# print(arr_2d.shape)  # 输出: (1, 136)

import threading
import queue
import time
import random

# 全局变量
食材队列 = queue.Queue()  # 学生将食材和号码牌放入队列
未吃饱学生数 = 5          # 初始未吃饱学生数
吃饱学生数 = 0            # 初始吃饱学生数
锁 = threading.Lock()     # 用于保护共享变量

# 学生类
class 学生(threading.Thread):
    def __init__(self, 编号):
        super().__init__()
        self.编号 = 编号
        self.是否吃饱 = False

    def 准备食材(self):
        time.sleep(random.uniform(0.5, 1.5))  # 模拟准备食材的时间
        食材 = f"食材-{self.编号}"
        print(f"学生 {self.编号} 准备好了 {食材}")
        return 食材

    def 吃饭(self):
        time.sleep(random.uniform(0.5, 1.5))  # 模拟吃饭的时间
        self.是否吃饱 = random.choice([True, False])  # 随机决定是否吃饱
        if self.是否吃饱:
            print(f"学生 {self.编号} 吃饱了！")
        else:
            print(f"学生 {self.编号} 还没吃饱，继续准备食材...")

    def run(self):
        global 未吃饱学生数, 吃饱学生数
        while not self.是否吃饱:
            # 准备食材
            食材 = self.准备食材()

            # 将食材和号码牌放入队列
            with 锁:
                食材队列.put((食材, self.编号))

            # 等待厨师通知
            while True:
                with 锁:
                    if 吃饱学生数 > 0:
                        吃饱学生数 -= 1
                        break
                time.sleep(0.1)  # 避免忙等待

            # 吃饭
            self.吃饭()

        # 学生吃饱后离开
        with 锁:
            未吃饱学生数 -= 1
        print(f"学生 {self.编号} 离开了。")

# 厨师类
class 厨师(threading.Thread):
    def __init__(self, 比例阈值):
        super().__init__()
        self.比例阈值 = 比例阈值

    def 处理食材(self, 食材列表):
        time.sleep(random.uniform(1, 2))  # 模拟处理食材的时间
        print(f"厨师处理了 {len(食材列表)} 份食材：{食材列表}")

    def run(self):
        global 未吃饱学生数, 吃饱学生数
        while 未吃饱学生数 > 0:
            # 检查队列长度是否满足条件
            with 锁:
                队列长度 = 食材队列.qsize()
                if 队列长度 / 未吃饱学生数 > self.比例阈值:
                    # 处理队列中所有食材
                    食材列表 = []
                    while not 食材队列.empty():
                        食材, 编号 = 食材队列.get()
                        食材列表.append((食材, 编号))

                    # 处理食材
                    self.处理食材(食材列表)

                    # 通知学生吃饭
                    with 锁:
                        吃饱学生数 += len(食材列表)

            time.sleep(0.1)  # 避免忙等待

        print("所有学生都吃饱了，厨师下班了！")

# 主程序
if __name__ == "__main__":
    # 创建学生线程
    学生列表 = []
    for i in range(5):
        学生线程 = 学生(i + 1)
        学生列表.append(学生线程)
        学生线程.start()

    # 创建厨师线程
    厨师线程 = 厨师(比例阈值=0.5)
    厨师线程.start()

    # 等待所有学生线程完成
    for 学生线程 in 学生列表:
        学生线程.join()

    # 等待厨师线程完成
    厨师线程.join()

    print("程序结束。")