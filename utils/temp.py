import threading
import queue
import time

# 生产者函数
def producer(q, result_q, condition):
    while True:
        # 生产商品
        item = produce_item()
        q.put(item)
        print(f"Producer {threading.get_ident()} produced {item}")
        # 通知消费者
        condition.acquire()
        condition.notify_all()
        condition.release()
        time.sleep(1)

# 消费者函数
def consumer(q, result_q, condition):
    while True:
        # 等待队列中有商品
        condition.acquire()
        while q.empty():
            condition.wait()
        item = q.get()
        condition.release()
        # 消费商品
        result = consume_item(item)
        result_q.put(result)
        print(f"Consumer {threading.get_ident()} consumed {item} and produced result {result}")
        q.task_done()
        time.sleep(1)

# 模拟生产商品
def produce_item():
    return f"item_{int(time.time())}"

# 模拟消费商品
def consume_item(item):
    return f"result_{item}"

# 创建队列
q = queue.Queue()
result_q = queue.Queue()
condition = threading.Condition()

# 创建生产者线程
producer_thread = threading.Thread(target=producer, args=(q, result_q, condition))
producer_thread.start()

# 创建消费者线程
consumer_thread = threading.Thread(target=consumer, args=(q, result_q, condition))
consumer_thread.start()

# 等待生产者和消费者线程结束
producer_thread.join()
consumer_thread.join()