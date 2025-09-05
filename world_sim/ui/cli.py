#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.console import Group
from rich.layout import Layout
from rich.table import Table
from rich.progress import ProgressBar
from rich.align import Align
from rich.rule import Rule
from rich import box
from rich.style import Style

from ..core import World
from ..core.entities import Creature
from ..logging import WorldLogger


@dataclass
class SimulationConfig:
    world_size: int = 20
    initial_creatures: int = 8
    max_ticks: int = 10_000
    food_spawn_interval: int = 3
    food_per_spawn: int = 5
    food_expire: int = 20
    mutation_rate: float = 0.08
    refresh_rate: int = 3
    seed: int = 42


class SimulationController:
    def __init__(self) -> None:
        self.console = Console()
        self.config = SimulationConfig()
        self.world: Optional[World] = None
        self.logger: Optional[WorldLogger] = None
        self.is_running = False
        self.layout: Optional[Layout] = None
        self._layout_constructed = False
        # Visual theme
        self._color_empty = (18, 18, 22)
        self._color_food = (255, 204, 0)
        self._default_creature = (56, 255, 146)
        self._cell_pixels = 2  # width per cell using background color blocks

    def _show_welcome(self) -> None:
        self.console.clear()
        header = Text()
        header.append("WORLD SIMULATION", style="bold cyan")
        header.append("\nAn advanced artificial life simulation\n", style="white")
        panel = Panel(header, box=box.DOUBLE, border_style="cyan")
        self.console.print(panel)

    def _initialize(self) -> None:
        self.logger = WorldLogger()
        self.world = World(size=self.config.world_size, rng_seed=self.config.seed, logger=self.logger)
        rng = self.world.rng
        for i in range(self.config.initial_creatures):
            x = int(rng.integers(0, self.world.size))
            y = int(rng.integers(0, self.world.size))
            c = Creature(x, y, energy=15.0)
            c.lineage = i
            self.world.add_creature(c)
        self.logger.log_event("World initialized", f"Size: {self.world.size}x{self.world.size}")
        self.logger.log_event("Creatures created", f"Count: {self.config.initial_creatures}")

    def _render_status(self, tick: int) -> Panel:
        # Build or update full dashboard layout
        if not self._layout_constructed:
            self._build_layout()
        self._update_header(tick)
        self._update_grid()
        self._update_simulation_stats()
        self._update_population_chart()
        self._update_creature_details()
        self._update_performance_metrics()
        self._update_footer()
        return Panel(Text(""))  # unused; layout is used directly

    def _build_layout(self) -> None:
        layout = Layout()
        console_height = self.console.height
        # Header, main, footer
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", size=max(18, console_height - 10)),
            Layout(name="footer", size=7),
        )
        # Main: grid left, analytics right
        layout["main"].split_row(
            Layout(name="simulation", ratio=3),
            Layout(name="analytics", ratio=2),
        )
        # Simulation: grid + stats
        layout["simulation"].split_column(
            Layout(name="grid_area", ratio=3),
            Layout(name="sim_stats", ratio=1),
        )
        # Analytics: population chart, details, metrics
        layout["analytics"].split_column(
            Layout(name="population_chart", ratio=1),
            Layout(name="creature_details", ratio=1),
            Layout(name="performance_metrics", ratio=1),
        )
        self.layout = layout
        self._layout_constructed = True

    def _update_header(self, tick: int) -> None:
        if not self.world:
            title = Text("WORLD SIMULATION", style="bold bright_cyan")
            self.layout["header"].update(Panel(Align.center(title), box=box.DOUBLE, border_style="bright_cyan"))
            return
        runtime_seconds = (self.world.metrics["start_time"].now() - self.world.metrics["start_time"]).seconds if hasattr(self.world.metrics["start_time"], 'now') else 0
        header_content = Group(
            Align.center(Text("WORLD SIMULATION", style="bold bright_cyan")),
            Align.center(Text(f"Tick {tick:06d} • Population {len(self.world.creatures)} • Food {int((self.world.grid=='F').sum())}", style="white")),
            Rule(style="bright_black"),
        )
        self.layout["header"].update(Panel(header_content, border_style="bright_cyan", box=box.DOUBLE, padding=(0, 1)))

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> tuple:
        h = h % 1.0
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        if i % 6 == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        return int(r * 255), int(g * 255), int(b * 255)

    def _lineage_rgb(self, lineage: Optional[int]) -> tuple:
        if lineage is None:
            return self._default_creature
        hue = (hash(lineage) % 360) / 360.0
        return self._hsv_to_rgb(hue, 0.75, 1.0)

    def _build_grid_text(self) -> Text:
        text = Text()
        if self.world is None:
            text.append("World not initialized", style="red")
            return text

        size = self.world.size
        console_width = self.console.width
        console_height = self.console.height

        # Estimate available rows/cols for colored block cells (2 chars per cell)
        available_cols_chars = max(20, int(console_width * 0.55) - 8)
        max_cols = min(size, max(1, available_cols_chars // self._cell_pixels))
        available_rows = max(10, int((console_height - 10) * 0.65))
        max_rows = min(size, available_rows)

        # Render cells as colored rectangles using background colors for a modern look
        # Each cell is two spaces with a background color
        def rgb_style(rgb: tuple, scale: float = 1.0) -> Style:
            r = max(0, min(255, int(rgb[0] * scale)))
            g = max(0, min(255, int(rgb[1] * scale)))
            b = max(0, min(255, int(rgb[2] * scale)))
            return Style(bgcolor=f"rgb({r},{g},{b})")

        style_empty = rgb_style(self._color_empty)
        style_food = rgb_style(self._color_food)
        style_birth = rgb_style((255, 105, 180))  # hot pink for birth flash

        creature_positions = {(c.x, c.y) for c in self.world.creatures}

        for i in range(max_rows):
            for j in range(max_cols):
                if (i, j) in creature_positions:
                    # Compute creature color with energy-based brightness
                    # Find the creature
                    # Fallback to default color if not found
                    creature = next((c for c in self.world.creatures if (c.x, c.y) == (i, j)), None)
                    # Use default creature color (bright green) for consistency with legend
                    base_rgb = self._default_creature
                    energy = getattr(creature, 'energy', 15.0)
                    scale = min(1.0, max(0.35, energy / 30.0))
                    style = rgb_style(base_rgb, scale)
                elif self.world.grid[i, j] == 'F':
                    style = style_food
                elif self.world.grid[i, j] == 'B':
                    style = style_birth
                else:
                    style = style_empty
                text.append("  ", style=style)
            text.append("\n")

        # Truncation notice
        if max_rows < size or max_cols < size:
            text.append("[truncated to fit terminal]", style="bright_black")

        return text

    def _update_grid(self) -> None:
        grid_content = self._build_grid_text()
        legend = Text()
        legend.append("  ", style=Style(bgcolor=f"rgb({self._color_food[0]},{self._color_food[1]},{self._color_food[2]})"))
        legend.append(" Food   ")
        legend.append("  ", style=Style(bgcolor=f"rgb({self._default_creature[0]},{self._default_creature[1]},{self._default_creature[2]})"))
        legend.append(" Creature", style=Style(color="white"))
        grid_panel = Panel(
            Group(grid_content, Text(""), legend),
            title="Simulation Grid",
            border_style="bright_black",
            box=box.HEAVY,
            padding=(0, 1),
        )
        self.layout["grid_area"].update(grid_panel)

    def _update_simulation_stats(self) -> None:
        if not self.world:
            self.layout["sim_stats"].update(Panel(Text("No data"), title="Stats"))
            return
        table = Table(show_header=False, box=box.ROUNDED, show_lines=False, padding=0)
        table.add_column("Metric", style="bright_white")
        table.add_column("Value", style="bright_cyan")
        current_pop = len(self.world.creatures)
        max_pop = max(self.world.metrics.get("peak_population", 0), current_pop, 1)
        table.add_row("Population", str(current_pop))
        table.add_row("Peak", str(self.world.metrics.get("peak_population", 0)))
        table.add_row("Food", str(int((self.world.grid == 'F').sum())))
        energy_bar = ProgressBar(total=100, completed=min(100, int(sum(c.energy for c in self.world.creatures)/max(1,current_pop*1.5)*100))) if current_pop else ProgressBar(total=100, completed=0)
        table.add_row("Avg Energy", "")
        table.add_row("", energy_bar)
        self.layout["sim_stats"].update(Panel(table, title="Live Stats", border_style="bright_black", box=box.ROUNDED))

    def _update_population_chart(self) -> None:
        if not self.world or len(self.world.population_history) < 2:
            self.layout["population_chart"].update(Panel(Text("Collecting data...", style="bright_black"), title="Population"))
            return
        chart_height = 6
        max_points = 30
        data = self.world.population_history[-max_points:]
        max_pop = max(max(data), 1)
        chart_text = Text()
        for h in range(chart_height, 0, -1):
            threshold = (h / chart_height) * max_pop
            line = "".join("█" if v >= threshold else " " for v in data)
            chart_text.append(f"{int(threshold):2d} ", style="bright_blue")
            chart_text.append(line, style="bright_green")
            chart_text.append("\n")
        chart_text.append("   ", style="bright_blue")
        chart_text.append("─" * len(data), style="bright_black")
        self.layout["population_chart"].update(Panel(chart_text, title="Population Trend", border_style="bright_black", box=box.ROUNDED, padding=(0, 1)))

    def _update_creature_details(self) -> None:
        if not self.world or not self.world.creatures:
            self.layout["creature_details"].update(Panel(Text("No creatures alive", style="bright_red"), title="Top Creatures"))
            return
        top = sorted(self.world.creatures, key=lambda c: c.energy, reverse=True)[:3]
        details = []
        for c in top:
            t = Text()
            cid = self.world.logger.creature_ids.get(c, "?")
            t.append(f"#{cid} ({c.x},{c.y})\n", style="bright_cyan")
            t.append(f"E:{c.energy:.1f} A:{c.age} V:{c.traits.vision} S:{c.traits.speed} M:{c.traits.metabolism:.2f}\n", style="white")
            pct = min(100, int((c.energy / 150.0) * 100))
            details.append(t)
            details.append(ProgressBar(total=100, completed=pct, width=24))
            details.append(Text("\n"))
        self.layout["creature_details"].update(Panel(Group(*details), title="Top Creatures", border_style="bright_black", box=box.ROUNDED, padding=(0, 1)))

    def _update_performance_metrics(self) -> None:
        if not self.world:
            self.layout["performance_metrics"].update(Panel(Text("No data"), title="Performance"))
            return
        table = Table(show_header=False, box=box.ROUNDED, show_lines=False, padding=0)
        table.add_column("Metric", style="bright_white")
        table.add_column("Value", style="bright_cyan")
        # Rates per minute
        from datetime import datetime
        runtime_minutes = max(1e-6, (datetime.now() - self.world.metrics["start_time"]).total_seconds() / 60)
        birth_rate = self.world.metrics.get("births", 0) / runtime_minutes
        death_rate = self.world.metrics.get("deaths", 0) / runtime_minutes
        table.add_row("Birth Rate", f"{birth_rate:.1f}/min")
        table.add_row("Death Rate", f"{death_rate:.1f}/min")
        table.add_row("Total Born", str(self.world.metrics.get("births", 0)))
        table.add_row("Total Died", str(self.world.metrics.get("deaths", 0)))
        table.add_row("Peak Pop", str(self.world.metrics.get("peak_population", 0)))
        table.add_row("Evol Events", str(self.world.metrics.get("evolutionary_events", 0)))
        table.add_row("Food Spawned", str(self.world.metrics.get("total_food_spawned", 0)))
        self.layout["performance_metrics"].update(Panel(table, title="Performance", border_style="bright_black", box=box.ROUNDED))

    def _update_footer(self) -> None:
        if not self.world:
            self.layout["footer"].update(Panel(Text("No events yet", style="bright_black"), title="Recent Events", border_style="bright_black", box=box.ROUNDED))
            return
        events = self.world.logger.get_recent_events(6)
        t = Text()
        for e in events:
            dot_style = "bright_blue"
            ev = e.get("event", "").lower()
            if "died" in ev:
                dot_style = "bright_red"
            elif "reproduced" in ev:
                dot_style = "bright_green"
            elif "food" in ev:
                dot_style = "bright_yellow"
            t.append("● ", style=dot_style)
            t.append(f"{e.get('event','')}: ", style="white")
            t.append(f"{e.get('details','')}\n", style="bright_cyan")
        self.layout["footer"].update(Panel(t, title="Recent Events", border_style="bright_black", box=box.ROUNDED, padding=(0, 1)))

    def run(self) -> None:
        if sys.version_info < (3, 8):
            print("Error: Python 3.8 or higher is required")
            sys.exit(1)
        self._show_welcome()
        self._initialize()
        self.is_running = True
        try:
            # Initialize layout and live rendering
            self._render_status(0)
            with Live(self.layout, refresh_per_second=self.config.refresh_rate, console=self.console) as live:
                for tick in range(self.config.max_ticks):
                    if not self.is_running:
                        break
                    self.world.step(
                        tick,
                        food_spawn_interval=self.config.food_spawn_interval,
                        food_per_spawn=self.config.food_per_spawn,
                        food_expire=self.config.food_expire,
                        mutation_rate=self.config.mutation_rate,
                    )
                    self._render_status(tick)
                    live.update(self.layout)
                    if not self.world.creatures:
                        self.logger.log_event("Extinction", "All creatures died", tick)
                        break
                    time.sleep(1.0 / self.config.refresh_rate)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Simulation interrupted[/yellow]")


def main() -> None:
    SimulationController().run()


if __name__ == "__main__":
    main()


