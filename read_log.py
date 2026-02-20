"""Read the interaction log and print a summary table."""
import json, os

log_path = os.path.join("data", "interactions.log")
if not os.path.exists(log_path):
    print("No log file found.")
    exit()

with open(log_path, encoding="utf-8") as f:
    lines = f.readlines()

print(f"\nTotal interactions logged: {len(lines)}\n")
print(f"{'Backend':12} {'Intent':20} {'Agent':26} {'User Input'}")
print("-" * 100)
for line in lines[-25:]:
    d = json.loads(line)
    ui = d.get("user_input", "")[:55]
    print(f"{d['backend']:12} {d['intent']:20} {d['target_agent']:26} {ui}")
