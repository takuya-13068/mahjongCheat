import socket
import time
import random

HOST = '127.0.0.1'
PORT = 8009

client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
since = time.time()

while True:
    now = time.time()
    if now - since >= 0.02:
        result = "0 0"
        client.sendto(result.encode('utf-8'),(HOST,PORT))
        since = now

