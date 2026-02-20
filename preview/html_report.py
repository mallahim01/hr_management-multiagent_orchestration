"""
preview/html_report.py
───────────────────────
Generates a self-contained, styled HTML report of all HR data for the active user.
Opens automatically in the default browser.
"""

import os
import webbrowser
from datetime import datetime
from database.db import DatabaseManager


class HTMLReportGenerator:
    """Produces a styled HTML file summarising all HR data."""

    def __init__(self, db: DatabaseManager, user_id: int, output_path: str = "data/report.html") -> None:
        self.db = db
        self.user_id = user_id
        self.output_path = output_path

    def generate(self, open_browser: bool = True) -> str:
        """Build and write the HTML report. Returns the output file path."""
        user    = self.db.get_user(self.user_id) or {"name": "Unknown", "department": "–", "email": "–"}
        balance = self.db.get_leave_balance(self.user_id) or {"total_leaves": 0, "used_leaves": 0, "remaining_leaves": 0}
        leave_rows = self.db.get_leave_requests(self.user_id)
        hr_rows    = self.db.get_hr_requests(self.user_id)
        conv_rows  = self.db.get_all_messages(self.user_id)

        html = self._build_html(user, balance, leave_rows, hr_rows, conv_rows)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"[HTMLReport] Report written to {self.output_path}")
        if open_browser:
            webbrowser.open(f"file:///{os.path.abspath(self.output_path)}")

        return self.output_path

    def _build_html(self, user, balance, leave_rows, hr_rows, conv_rows) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        leave_html = self._rows_to_table(
            ["ID", "From", "To", "Reason", "Status", "Submitted"],
            [[f"LR-{r['id']:04d}", r["start_date"], r["end_date"],
              r["reason"], r["status"], r["created_at"][:10]] for r in leave_rows],
        )
        hr_html = self._rows_to_table(
            ["Ticket", "Request", "Date"],
            [[f"HR-{r['id']:04d}", r["request_text"], r["created_at"][:10]] for r in hr_rows],
        )
        conv_html = self._rows_to_table(
            ["Time", "Role", "Agent", "Intent", "Message"],
            [[r["timestamp"][:16], r["role"], r.get("agent") or "–",
              r.get("intent") or "–", r["content"][:120]] for r in conv_rows],
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HR Demo Report – {user['name']}</title>
<style>
  :root {{
    --bg: #0f1117; --card: #1a1d27; --accent: #7c6af7;
    --text: #e2e8f0; --muted: #8892a4; --border: #2d3148;
    --green: #22c55e; --yellow: #f59e0b; --red: #ef4444;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); padding: 2rem; }}
  h1 {{ font-size: 1.8rem; font-weight: 700; color: var(--accent); margin-bottom: .25rem; }}
  .meta {{ color: var(--muted); font-size: .875rem; margin-bottom: 2rem; }}
  .cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem; text-align: center; }}
  .card .val {{ font-size: 2.5rem; font-weight: 700; }}
  .card .lbl {{ color: var(--muted); font-size: .8rem; margin-top: .25rem; text-transform: uppercase; letter-spacing: .05em; }}
  .card.green .val {{ color: var(--green); }}
  .card.yellow .val {{ color: var(--yellow); }}
  .card.accent .val {{ color: var(--accent); }}
  section {{ margin-bottom: 2rem; }}
  h2 {{ font-size: 1.1rem; color: var(--accent); margin-bottom: 1rem; padding-bottom: .5rem; border-bottom: 1px solid var(--border); }}
  table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 10px; overflow: hidden; font-size: .875rem; }}
  th {{ background: var(--border); color: var(--muted); font-weight: 600; text-align: left; padding: .75rem 1rem; font-size: .75rem; text-transform: uppercase; letter-spacing: .05em; }}
  td {{ padding: .7rem 1rem; border-bottom: 1px solid var(--border); vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(124,106,247,.06); }}
  .badge {{ display: inline-block; padding: .2em .6em; border-radius: 9999px; font-size: .75rem; font-weight: 600; }}
  .badge-pending {{ background: #f59e0b22; color: var(--yellow); }}
  .badge-approved {{ background: #22c55e22; color: var(--green); }}
  .badge-user {{ background: #7c6af722; color: var(--accent); }}
  .badge-assistant {{ background: #22c55e22; color: var(--green); }}
  .empty {{ color: var(--muted); text-align: center; padding: 2rem; }}
</style>
</head>
<body>
<h1>🏢 ACME Corp – HR Demo Report</h1>
<p class="meta">Employee: <strong>{user['name']}</strong> | Department: {user['department']} | Generated: {now}</p>

<div class="cards">
  <div class="card accent"><div class="val">{balance['total_leaves']}</div><div class="lbl">Total Leave Days</div></div>
  <div class="card yellow"><div class="val">{balance['used_leaves']}</div><div class="lbl">Used</div></div>
  <div class="card green"><div class="val">{balance['remaining_leaves']}</div><div class="lbl">Remaining</div></div>
</div>

<section>
  <h2>📋 Leave Requests</h2>
  {"<p class='empty'>No leave requests yet.</p>" if not leave_rows else leave_html}
</section>

<section>
  <h2>📨 HR Requests</h2>
  {"<p class='empty'>No HR requests yet.</p>" if not hr_rows else hr_html}
</section>

<section>
  <h2>💬 Conversation History</h2>
  {"<p class='empty'>No conversations yet.</p>" if not conv_rows else conv_html}
</section>
</body>
</html>"""

    @staticmethod
    def _rows_to_table(headers: list, rows: list) -> str:
        th = "".join(f"<th>{h}</th>" for h in headers)
        trs = ""
        for row in rows:
            cells = ""
            for i, cell in enumerate(row):
                val = str(cell)
                if headers[i] == "Status":
                    cls = "badge-approved" if val.lower() == "approved" else "badge-pending"
                    val = f'<span class="badge {cls}">{val}</span>'
                elif headers[i] == "Role":
                    cls = "badge-user" if val == "user" else "badge-assistant"
                    val = f'<span class="badge {cls}">{val}</span>'
                cells += f"<td>{val}</td>"
            trs += f"<tr>{cells}</tr>"
        return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"
