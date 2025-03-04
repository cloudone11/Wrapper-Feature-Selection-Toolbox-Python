import asyncio
import asyncssh

async def connect():
    async with asyncssh.connect('localhost', port=8022, username='guest', password='') as conn:
        print(await conn.run('echo "Hello, server!"').stdout)

asyncio.run(connect())