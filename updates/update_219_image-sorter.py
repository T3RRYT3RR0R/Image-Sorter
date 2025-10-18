# ensure network connectivity before script run.  
  #  Update file
# key: f6819f209bac39679b31dd48bd92abcb96b9c65bc8ffbca7f1f0e9743bb4b8ef

import socket
def check_internet(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

no_internet = True

while no_internet:
    if not check_internet():
        input("Warning - Internet connection required. Connect to the internet to continue.")
    else:
        no_internet = False
        continue
#‍ ∆eof
