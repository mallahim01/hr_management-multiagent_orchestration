"""Read the interaction log and print a summary table."""
import json, os

log_path = os.path.join("data", "interactions.log")
if not os.path.exists(log_path):
    print("No log file found.")
    exit()

with open(log_path, encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

# Records written before the "event" field existed are all interactions.
turns  = [r for r in records if r.get("event", "interaction") == "interaction"]
events = [r for r in records if r.get("event", "interaction") != "interaction"]

print(f"\nTotal interactions logged: {len(turns)}\n")
print(f"{'Backend':12} {'Intent':20} {'Agent':26} {'User Input'}")
print("-" * 100)
for d in turns[-25:]:
    ui = (d.get("user_input") or "")[:55]
    print(f"{d.get('backend',''):12} {d.get('intent',''):20} {d.get('target_agent',''):26} {ui}")

if events:
    print(f"\nDomain events: {len(events)}\n")
    print(f"{'Event':28} {'Reason':26} {'Detail'}")
    print("-" * 100)
    for d in events[-25:]:
        detail = (d.get("detail") or "")[:44]
        print(f"{d.get('event',''):28} {d.get('reason_code',''):26} {detail}")
