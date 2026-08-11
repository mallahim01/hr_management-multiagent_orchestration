"""Read the interaction log and print a summary table."""
import json, os, sys

from core.logger import DEFAULT_LOG_PATH

log_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG_PATH
if not os.path.exists(log_path):
    print(f"No log file at {log_path}. Run the app first, or try:")
    print("  python read_log.py logs/sample-session.log")
    raise SystemExit(1)

with open(log_path, encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

# Records written before the "event" field existed are all interactions.
turns  = [r for r in records if r.get("event", "interaction") == "interaction"]
events = [r for r in records if r.get("event", "interaction") != "interaction"]

print(f"\n{log_path}: {len(turns)} turns, {len(events)} domain events\n")
print(f"{'Backend':12} {'Intent':20} {'Agent':26} {'User Input'}")
print("-" * 100)
for d in turns[-25:]:
    ui = (d.get("user_input") or "")[:55]
    print(f"{d.get('backend',''):12} {d.get('intent',''):20} {d.get('target_agent',''):26} {ui}")

if events:
    print(f"\nDomain events (last 25 of {len(events)})\n")
    print(f"{'Event':30} {'Reason':26} {'Detail'}")
    print("-" * 100)
    for d in events[-25:]:
        detail = str(d.get("detail") or d.get("source") or "")[:42]
        print(f"{d.get('event',''):30} {d.get('reason_code',''):26} {detail}")
