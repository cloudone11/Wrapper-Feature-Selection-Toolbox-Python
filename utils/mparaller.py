import threading
import time
import random

# 全局变量
thread_storage = [None] * 6
result_list = [0] * 6
result_used_flag = [False] * 6
barrier = threading.Barrier(7)  # 6 个 worker 线程 + 1 个 processor 线程
event = threading.Event()
stop_flag = False
max_iterater = 2
def worker_thread(index):
    while True:
        global stop_flag, thread_storage, result_list, result_used_flag
        sleep_time = random.uniform(0, 1)
        time.sleep(sleep_time)
        thread_storage[index] = sleep_time
        print(f'Worker {index} sleep {sleep_time} seconds')
        print(f'reached barrier {index}')
        barrier.wait()  # 等待所有线程到达屏障
        print(f'passed barrier {index}')
        event.wait()  # 等待处理完成
        print(f"Result for worker {index}: {result_list[index]}")
        result_used_flag[index] = True
        if all(result_used_flag):
            print("All results are used")
            event.clear()  # 重置事件
        if stop_flag:
            break

def processing_thread():
    print("Processing thread start waiting")
    while True:
        global max_iterater , stop_flag, thread_storage, result_list, result
        print("Processing thread waiting")
        barrier.wait()  # 等待所有 worker 线程到达屏障
        print("Processing thread start")
        print(f"Thread storage: {thread_storage}")
        # 处理 thread_storage
        for i in range(len(thread_storage)):
            result_list[i] = thread_storage[i] * 2  # 示例处理：将存储的值乘以 2
            result_used_flag[i] = False  # 重置 result_used_flag
            thread_storage[i] = None  # 清空 thread_storage
        print("Processing thread finish") 
        
        max_iterater -= 1
        if max_iterater == 0:
            stop_flag = True
        event.set()  # 释放完成事件
        if stop_flag:
            break

def observer_thread():
    while True:
        print(f"Thread storage: {thread_storage}")
        print(f"Result list: {result_list}")
        print(f"Result used flag: {result_used_flag}")
        if stop_flag:
            break
        time.sleep(5)

# 创建并启动工作线程
for i in range(6):
    t = threading.Thread(target=worker_thread, args=(i,))
    t.start()
    print(f"Worker {i} started")
print("All worker threads started")
# 创建并启动处理线程
processing_t = threading.Thread(target=processing_thread)
processing_t.start()
print("Processing thread started")
# 创建并启动观察线程
observer_t = threading.Thread(target=observer_thread)
observer_t.start()

# 等待所有工作线程完成
for i in range(6):
    t.join()
    print(f"Worker {i} joined")

# 等待处理线程完成（可以设置一个条件来终止处理线程）
processing_t.join()
print("Processing thread joined")
# 等待观察线程完成（可以设置一个条件来终止观察线程）
observer_t.join()
print("Observer thread joined")