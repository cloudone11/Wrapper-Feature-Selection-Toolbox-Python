import redis

# 连接到 Redis 服务器
r = redis.Redis(host='localhost', port=6379, db=0)

# 字符串操作
r.set('foo', 'bar')
print(r.get('foo'))  # 输出: b'bar'

# 哈希操作
r.hset('myhash', 'field1', 'value1')
print(r.hget('myhash', 'field1'))  # 输出: b'value1'

# 列表操作
r.lpush('mylist', 'element1')
r.lpush('mylist', 'element2')
print(r.lrange('mylist', 0, -1))  # 输出: [b'element2', b'element1']

# 集合操作
r.sadd('myset', 'member1')
r.sadd('myset', 'member2')
print(r.smembers('myset'))  # 输出: {b'member1', b'member2'}

# 有序集合操作
r.zadd('myzset', {'member1': 1, 'member2': 2})
print(r.zrange('myzset', 0, -1, withscores=True))  # 输出: [(b'member1', 1.0), (b'member2', 2.0)]

# 发布/订阅
def message_handler(message):
    print(f"Received message: {message['data']}")

p = r.pubsub()
p.subscribe(**{'my-channel': message_handler})
p.run_in_thread(sleep_time=0.001)

r.publish('my-channel', 'Hello, Redis!')