"""Drives the EdgeHand nodes against fake_cell.py and checks the results. Run via run_integration.sh."""
import json, socket, sys, time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String


class Check(Node):
    def __init__(self):
        super().__init__("integration_check")
        self.states, self.says, self.cell = [], [], []
        self.create_subscription(String, "/edgehand/state", lambda m: self.states.append(m.data), 50)
        self.create_subscription(String, "/edgehand/say", lambda m: self.says.append(m.data), 50)
        self.create_subscription(String, "/test/log", lambda m: self.cell.append(m.data), 50)
        self.cmd = self.create_publisher(String, "/edgehand/command", 10)
        self.req = self.create_publisher(String, "/edgehand/tool_request", 10)
        self.pull = self.create_publisher(Empty, "/test/pull", 10)
        self.stop = self.create_publisher(Empty, "/edgehand/stop", 10)

    def spin_until(self, cond, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if cond():
                return True
        return False

    def reset(self):
        self.states.clear(), self.says.clear(), self.cell.clear()


def udp(kind, port=5005):
    socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(json.dumps({"type": kind}).encode(), ("127.0.0.1", port))


def main():
    rclpy.init()
    c = Check()
    results = {}
    c.spin_until(lambda: False, 3.0)                              # let discovery settle

    # 1. text -> Gemma -> vision -> MoveIt -> tug -> release
    c.reset()
    c.cmd.publish(String(data="could you pass me the cross-head screwdriver"))
    held = c.spin_until(lambda: any(s.startswith("holding") for s in c.states), 90)
    if held:
        c.pull.publish(Empty())
    done = c.spin_until(lambda: "idle" in c.states[1:] and any("Here's the phillips" in s for s in c.says), 30)
    results["text request -> handover"] = held and done
    print("--- scenario 1 states:", [s for s in c.states if not s.startswith("holding: delta")][:12])
    print("--- scenario 1 says:", c.says)
    print("--- scenario 1 cell log:\n   " + "\n   ".join(c.cell))

    # 2. STOP while moving cancels the trajectory and never opens the gripper afterwards
    c.reset()
    time.sleep(1.5)                                               # handover_node refuses requests right after a stop
    c.req.publish(String(data="phillips"))
    moving = c.spin_until(lambda: "picking" in c.states, 30)
    c.spin_until(lambda: False, 0.4)
    c.stop.publish(Empty())
    stopped = c.spin_until(lambda: "stopped" in c.states, 10)
    results["STOP cancels motion"] = moving and stopped and any("CANCELED" in s for s in c.cell)
    print("--- scenario 2 states:", c.states, "\n--- cell:", c.cell[-4:])

    # 3. Pi buttons over UDP: REQUEST then RELEASE
    c.reset()
    time.sleep(1.5)
    udp("request")
    held = c.spin_until(lambda: any(s.startswith("holding") for s in c.states), 30)
    udp("release")
    done = c.spin_until(lambda: "idle" in c.states[1:], 30)
    results["Pi REQUEST + RELEASE over UDP"] = held and done
    print("--- scenario 3 states:", [s for s in c.states if not s.startswith("holding: delta")])

    print("\n".join(f"{'PASS' if ok else 'FAIL'}  {name}" for name, ok in results.items()))
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
