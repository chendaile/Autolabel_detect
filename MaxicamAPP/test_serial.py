import struct, serial
import time

ser = serial.Serial('/dev/serial0', 115200)
data_list = [0, 0, 0]
last_num = sum([0xAA,0x06,*data_list]) % (0xFF + 1)
data =  struct.pack("<"+"B"*6,0xAA,0x06,*data_list,last_num)
while True:
    ser.write(data)
    print(f"sucessful send {data_list}")
    time.sleep(1)