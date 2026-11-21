from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class TerminalUI:
    """
    Terminal UI for displaying the Plancraft environment using rich.

    Creates a fixed UI with three main areas:
    - Grid: Displays the current state of the environment
    - Logs: Shows the agent's logs in a scrolling area
    - Stats: Shows statistics about the current evaluation run
    """

    def __init__(self, title: str = "Plancraft Environment"):
        self.console = Console()
        self.layout = Layout()
        self.title = title
        self.logs = []  # Will store rich Text objects instead of plain strings
        self.max_logs = 20  # Maximum number of log lines to show
        self.stats = {
            "success_rate": 0.0,
            "examples_completed": 0,
            "total_examples": 0,
            "current_example": "",
            "steps": 0,
            "memory_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "tokens_used_memory": 0,
            "tokens_used_agent": 0,
            "tokens_used": 0,
        }

        # Create the layout
        self._setup_layout()

        # Live display
        self.live = Live(self.layout, refresh_per_second=2)

    def _setup_layout(self) -> None:
        """Set up the layout for the terminal UI."""
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer"),
        )
        self.layout["main"].split_row(
            Layout(name="grid", ratio=2),
            Layout(name="stats", ratio=1),
        )
        self.layout["footer"].update(Panel(Text("Logs"), title="Agent Logs"))
        self.layout["header"].update(
            Panel(Text(f"{self.title}", justify="center"), padding=0)
        )

    def _generate_grid_panel(self, observation: dict[str, dict]) -> Panel:
        if "inventory" not in observation:
            return Panel(Text("No grid available"), title="Environment")

        inventory = observation["inventory"]
        target = observation.get("target", "Unknown")

        # Create tables for different inventory sections
        crafting_grid = Table.grid(padding=(0, 2), expand=True)
        inventory_grid = Table.grid(padding=(0, 2), expand=True)
        crafting_slot = Table.grid(padding=(0, 2), expand=True)

        # Create a container for all grids
        main_grid = Table.grid(expand=True)

        # Add crafting slot (index 0)
        craft_result = inventory.get(0, {"type": "empty", "quantity": 0})
        crafting_slot.add_row(
            f"[bold]{craft_result['type']}[/bold]\n({craft_result['quantity']})"
        )

        # Add crafting grid (index 1-9, 3x3 grid)
        for row in range(3):
            crafting_row = []
            for col in range(3):
                slot_idx = 1 + row * 3 + col
                item = inventory.get(slot_idx, {"type": "empty", "quantity": 0})
                crafting_row.append(
                    f"[bold]{item['type']}[/bold]\n({item['quantity']})"
                )
            crafting_grid.add_row(*crafting_row)

        # Add inventory grid (index 10-36, typically 3x9 grid but we'll use 4x9)
        for row in range(3):  # Main inventory
            inv_row = []
            for col in range(9):
                slot_idx = 10 + row * 9 + col
                item = inventory.get(slot_idx, {"type": "empty", "quantity": 0})
                inv_row.append(f"[bold]{item['type']}[/bold]\n({item['quantity']})")
            inventory_grid.add_row(*inv_row)

        # Add hotbar (last row of inventory)
        hotbar_row = []
        for col in range(9):
            slot_idx = 37 + col  # Assume hotbar is after main inventory
            item = inventory.get(slot_idx, {"type": "empty", "quantity": 0})
            hotbar_row.append(f"[bold]{item['type']}[/bold]\n({item['quantity']})")
        inventory_grid.add_row(*hotbar_row)

        # Create a top row with crafting grid and crafting result
        top_container = Table.grid(expand=True)
        top_container.add_row(
            Panel(crafting_grid, title="Crafting Grid"),
            Panel(crafting_slot, title="Result"),
        )

        # Add all components to the main grid
        main_grid.add_row(top_container)
        main_grid.add_row(Panel(inventory_grid, title="Inventory"))

        # Return the complete panel
        return Panel(main_grid, title=f"Environment (Target: {target})")

    def _generate_stats_panel(self) -> Panel:
        stats_table = Table.grid(expand=True)
        stats_table.add_row("Success Rate:", f"{self.stats['success_rate']:.2%}")
        stats_table.add_row(
            "Examples Completed:", str(self.stats["examples_completed"])
        )
        stats_table.add_row("Total Examples:", str(self.stats["total_examples"]))
        stats_table.add_row("Current Example:", self.stats["current_example"])
        stats_table.add_row(
            "Is Impossible:", str(self.stats.get("is_impossible", False))
        )
        stats_table.add_row("Steps:", str(self.stats["steps"]))

        stats_table.add_row("Memory Calls:", str(self.stats["memory_calls"]))
        stats_table.add_row("Cache Hits:", str(self.stats["cache_hits"]))
        stats_table.add_row("Cache Misses:", str(self.stats["cache_misses"]))
        stats_table.add_row(
            "Tokens Used (Memory):", str(self.stats["tokens_used_memory"])
        )
        stats_table.add_row(
            "Tokens Used (Agent):", str(self.stats["tokens_used_agent"])
        )
        stats_table.add_row("Total Tokens Used:", str(self.stats["tokens_used"]))

        return Panel(stats_table, title="Statistics")

    def _generate_logs_panel(self) -> Panel:
        # Instead of joining strings, we'll combine Text objects
        if not self.logs:
            return Panel(Text("No logs yet..."), title="Agent Logs")

        # Take the most recent logs and reverse them (newest first)
        recent_logs = self.logs[-self.max_logs :][::-1]

        # Create a container for the logs
        log_container = Table.grid(expand=True)

        for log in recent_logs:
            log_container.add_row(log)

        return Panel(log_container, title="Agent Logs")

    def update_grid(self, observation: dict[str, Any]) -> None:
        grid_panel = self._generate_grid_panel(observation)
        self.layout["grid"].update(grid_panel)

    def update_stats(self, stats: dict[str, Any]) -> None:
        self.stats.update(stats)
        stats_panel = self._generate_stats_panel()
        self.layout["stats"].update(stats_panel)
        # reset logs
        if "success_rate" in stats:
            self.logs = []
            self.layout["footer"].update(self._generate_logs_panel())

    def increment_stats(self, key: str, value: int) -> None:
        """Increment a specific stat by a given value."""
        if key in self.stats:
            self.stats[key] += value
            self.layout["stats"].update(self._generate_stats_panel())
        else:
            raise KeyError(f"Stat '{key}' not found in stats dictionary.")

    def add_log(self, message: Text) -> None:
        """Add a rich Text log message to the logs."""
        self.logs.append(message)
        self.layout["footer"].update(self._generate_logs_panel())

    def start(self) -> None:
        """Start the live display."""
        self.live.start()

    def stop(self) -> None:
        """Stop the live display."""
        self.live.stop()

    def __enter__(self):
        """Context manager enter method."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method."""
        self.stop()


# Create a custom Loguru sink to redirect logs to the terminal UI
class TerminalUISink:
    """
    Custom sink for loguru that redirects logs to the terminal UI.
    """

    def __init__(self, terminal_ui: TerminalUI):
        self.terminal_ui = terminal_ui
        # Color mapping for log levels
        self.level_colors = {
            "TRACE": "dim blue",
            "DEBUG": "blue",
            "INFO": "green",
            "SUCCESS": "bold green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold white on red",
        }

    def __call__(self, message):
        record = message.record
        level = record["level"].name
        msg = record["message"]

        # Create a rich Text object with appropriate styling based on log level
        level_color = self.level_colors.get(level, "white")
        log_text = Text()
        log_text.append(f"[{level}] ", style=level_color)

        # If the message contains any markdown-like formatting, try to preserve it
        if any(marker in msg for marker in ["*", "_", "`", "["]):
            # Handle any code blocks or formatting in the message
            try:
                # Use plain text but preserve basic formatting like code blocks
                log_text.append(msg)
            except:
                # Fallback to plain text if parsing fails
                log_text.append(msg)
        else:
            # Plain text message
            log_text.append(msg)

        self.terminal_ui.add_log(log_text)
