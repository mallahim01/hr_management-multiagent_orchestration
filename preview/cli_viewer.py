"""
preview/cli_viewer.py
──────────────────────
Prints leave requests, HR requests, and leave balances as formatted tables
in the terminal using rich + tabulate.
"""

from typing import Any, Dict, List
from rich.console import Console
from rich.table import Table
from rich import box

from database.db import DatabaseManager

console = Console()


class CLIViewer:
    """Renders HR data as styled terminal tables."""

    def __init__(self, db: DatabaseManager, user_id: int) -> None:
        self.db = db
        self.user_id = user_id

    def show_all(self) -> None:
        """Print all three data views."""
        user = self.db.get_user(self.user_id)
        name = user["name"] if user else f"User {self.user_id}"

        console.print(f"\n[bold cyan]━━━ HR Preview for {name} (ID: {self.user_id}) ━━━[/bold cyan]\n")
        self.show_leave_balance()
        self.show_leave_requests()
        self.show_hr_requests()
        self.show_recent_conversations()

    def show_leave_balance(self) -> None:
        balance = self.db.get_leave_balance(self.user_id)
        table = Table(title="🌴 Leave Balance", box=box.ROUNDED, style="cyan")
        table.add_column("Total",     style="white")
        table.add_column("Used",      style="yellow")
        table.add_column("Remaining", style="green bold")
        if balance:
            table.add_row(
                str(balance["total_leaves"]),
                str(balance["used_leaves"]),
                str(balance["remaining_leaves"]),
            )
        else:
            table.add_row("–", "–", "–")
        console.print(table)

    def show_leave_requests(self) -> None:
        rows = self.db.get_leave_requests(self.user_id)
        table = Table(title="📋 Leave Requests", box=box.ROUNDED, style="green")
        table.add_column("ID",    style="bold")
        table.add_column("From",  style="cyan")
        table.add_column("To",    style="cyan")
        table.add_column("Reason")
        table.add_column("Status", style="yellow")
        table.add_column("Submitted")
        for r in rows:
            table.add_row(
                f"LR-{r['id']:04d}",
                r["start_date"],
                r["end_date"],
                r["reason"],
                r["status"],
                r["created_at"][:10],
            )
        if not rows:
            table.add_row("–", "–", "–", "No requests yet", "–", "–")
        console.print(table)

    def show_hr_requests(self) -> None:
        rows = self.db.get_hr_requests(self.user_id)
        table = Table(title="📨 HR Requests", box=box.ROUNDED, style="magenta")
        table.add_column("Ticket",  style="bold")
        table.add_column("Request", no_wrap=False, max_width=60)
        table.add_column("Date",    style="cyan")
        for r in rows:
            table.add_row(
                f"HR-{r['id']:04d}",
                r["request_text"],
                r["created_at"][:10],
            )
        if not rows:
            table.add_row("–", "No HR requests yet", "–")
        console.print(table)

    def show_recent_conversations(self) -> None:
        rows = self.db.get_all_messages(self.user_id)
        table = Table(title="💬 Recent Conversations (last 50)", box=box.ROUNDED, style="blue")
        table.add_column("Time",    style="dim", width=20)
        table.add_column("Role",    style="bold", width=10)
        table.add_column("Agent",   style="cyan", width=22)
        table.add_column("Intent",  style="yellow", width=18)
        table.add_column("Message", no_wrap=False, max_width=55)
        for r in rows:
            table.add_row(
                r["timestamp"][:16],
                r["role"],
                r.get("agent") or "–",
                r.get("intent") or "–",
                r["content"][:100],
            )
        if not rows:
            table.add_row("–", "–", "–", "–", "No conversation history yet")
        console.print(table)
