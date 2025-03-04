# To run this program, the file ``ssh_host_key`` must exist with an SSH
# private key in it to use as a server host key. An SSH host certificate
# can optionally be provided in the file ``ssh_host_key-cert.pub``.

import asyncio, asyncssh, bcrypt, sys
from typing import Optional

passwords = {'guest': b'',                # guest account with no password
             'user123': bcrypt.hashpw(b'secretpw', bcrypt.gensalt()),
            }

def handle_client(process: asyncssh.SSHServerProcess) -> None:
    username = process.get_extra_info('username')
    process.stdout.write(f'Welcome to my SSH server, {username}!\n')
    process.exit(0)

class MySSHServer(asyncssh.SSHServer):
    def connection_made(self, conn: asyncssh.SSHServerConnection) -> None:
        peername = conn.get_extra_info('peername')[0]
        print(f'SSH connection received from {peername}.')

    def connection_lost(self, exc: Optional[Exception]) -> None:
        if exc:
            print('SSH connection error: ' + str(exc), file=sys.stderr)
        else:
            print('SSH connection closed.')

    def begin_auth(self, username: str) -> bool:
        # If the user's password is the empty string, no auth is required
        return passwords.get(username) != b''

    def password_auth_supported(self) -> bool:
        return True

    def validate_password(self, username: str, password: str) -> bool:
        if username not in passwords:
            return False
        pw = passwords[username]
        if not password and not pw:
            return True
        return bcrypt.checkpw(password.encode('utf-8'), pw)

async def start_server() -> None:
    await asyncssh.create_server(MySSHServer, '', 8022,
                                 server_host_keys=['ssh_host_key'],
                                 process_factory=handle_client)

loop = asyncio.new_event_loop()

try:
    loop.run_until_complete(start_server())
except (OSError, asyncssh.Error) as exc:
    sys.exit('Error starting server: ' + str(exc))

loop.run_forever()