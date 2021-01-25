import time
import subprocess
import locale
import codecs

mylist = []
ps = subprocess.Popen('ipconfig', stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
while True:
    data = ps.stdout.readline()
    print(data.decode())
    if data == b'':
        if ps.poll() is not None:
            break
    else:
        mylist.append(data.decode(codecs.lookup(locale.getpreferredencoding()).name))
        newlist = []
        for i in mylist:
            if i.find('192.168') > 0:
                newlist.append(i)
        newlist.sort()
        print('Sum of requests from LAN:', len(newlist))