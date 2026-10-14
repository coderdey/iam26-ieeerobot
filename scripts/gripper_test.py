"""Find gripper positions for each object. Run on desktop with arm connected."""

from scservo_sdk import PortHandler, PacketHandler
import time

PORT = "/dev/ttyACM1"       # adjust to your follower arm port
BAUDRATE = 1_000_000
GRIPPER_ID = 6

ph = PortHandler(PORT)
ph.openPort()
ph.setBaudRate(BAUDRATE)
pk = PacketHandler(0)

pk.write2ByteTxRx(ph, GRIPPER_ID, 46, 100)  # slow speed

print("Type a position (1400=fully open, 5000=fully closed) or 'q' to quit")
print("Start at 1400, increase slowly until object is gripped firmly")
print("STOP BEFORE THE SERVO STALLS — if it buzzes, go back!\n")

while True:
    val = input("Position > ")
    if val.strip().lower() == 'q':
        break
    try:
        pos = int(val)
        pk.write2ByteTxRx(ph, GRIPPER_ID, 42, pos)
        print(f"  Moved gripper to {pos}")
        time.sleep(0.5)
        cur_pos, _, _ = pk.read2ByteTxRx(ph, GRIPPER_ID, 56)
        print(f"  Actual position: {cur_pos}")
    except ValueError:
        print("  Enter a number or 'q'")

pk.write2ByteTxRx(ph, GRIPPER_ID, 42, 1400)  # open before exit
ph.closePort()
print("Done.")

