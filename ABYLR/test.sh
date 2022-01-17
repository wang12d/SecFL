#! /bin/bash
cd /home/flyingtom/ABY-LR
/usr/bin/python3 server.py > log/server.log 2>&1
/usr/bin/python3 client.py > log/client.log 2>&1