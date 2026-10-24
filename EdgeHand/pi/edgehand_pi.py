"""Raspberry Pi 4: three buttons + a heartbeat, sent to the workstation over UDP. No ROS on the Pi.

    python3 edgehand_pi.py --host 192.168.60.1
    python3 edgehand_pi.py --host 192.168.60.1 --keys     # no buttons: type r / b / s + ENTER

Wiring (each button between the GPIO pin and GND): REQUEST GPIO17, RELEASE GPIO27, STOP GPIO22.
This is a stop REQUEST, not an emergency stop. The Kinova's red E-stop is the e-stop.
"""
import argparse, json, socket, threading, time

PINS = {"request": 17, "release": 27, "stop": 22}
HEARTBEAT_HZ = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.60.1")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--keys", action="store_true", help="keyboard instead of GPIO")
    a = ap.parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = [0]

    def send(kind):
        seq[0] += 1
        sock.sendto(json.dumps({"type": kind, "seq": seq[0], "t": time.time()}).encode(), (a.host, a.port))
        if kind != "heartbeat":
            print("sent", kind)

    def heartbeat():
        while True:
            send("heartbeat")
            time.sleep(1 / HEARTBEAT_HZ)

    threading.Thread(target=heartbeat, daemon=True).start()
    if a.keys:
        keys = {"r": "request", "b": "release", "s": "stop"}
        print("r = request, b = release, s = stop")
        while True:
            k = input().strip().lower()[:1]
            if k in keys:
                send(keys[k])
    else:
        from gpiozero import Button
        buttons = []
        for kind, pin in PINS.items():
            b = Button(pin, pull_up=True, bounce_time=0.05)
            b.when_pressed = lambda kind=kind: send(kind)
            buttons.append(b)
        print("buttons ready:", PINS)
        threading.Event().wait()


if __name__ == "__main__":
    main()
