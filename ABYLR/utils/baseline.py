import socket
from utils.common import Role

port = [8082,8083]

def sendby(role,data):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while(True):
        try:
            client.connect(("localhost", port[role]))
            break
        except:
            continue
        
    client.sendall(data.encode("utf-8"))
    client.close()



def recvby(role):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while(True):
        try:
            server.bind(("localhost", port[(int)(not role)]))
            break
        except:
            continue
    server.listen()
    conn, _ = server.accept()

    total_data = str()
    data = conn.recv(1024).decode("utf-8")
    total_data += data

    while len(data)>0:
        data = conn.recv(1024).decode("utf-8")
        total_data += data 

    server.close()
    return total_data


def communicate(role, data):
    data = str(data)
    if(role == Role.SERVER):
        other_data = recvby(role)
        sendby(role, data)
    else:
        sendby(role, data)
        other_data = recvby(role)
    return other_data

