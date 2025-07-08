import subprocess
import time
import sys

server = subprocess.Popen(["build\\Debug\\cursor_server.exe"])
time.sleep(1)

client = subprocess.Popen([sys.executable,"client.py"])
client.wait()

server.terminate()
server.wait()