from typing import List, Dict, Tuple
import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.box import HEAVY, ROUNDED, DOUBLE
from rich.style import Style
from rich.progress import ProgressBar
from rich.align import Align
from rich.rule import Rule
from rich.columns import Columns
from rich.console import Group
from datetime import datetime
from creature import Creature
from World_Logger import WorldLogger

class World:
    def __init__(self, size: int = 50, rng_seed: int = None, logger: WorldLogger = None):
        self.size = size
        self.grid = np.full((size, size), None, dtype=object)
        self.food_spawn_times = {}
        self.rng = np.random.default_rng(rng_seed)
        self.logger = logger or WorldLogger()
        self.creatures: List[Creature] = []
        self.console = Console()
        self.layout = Layout()
        # Will be filled dynamically based on layout sizing to help grid alignment
        self._sim_panel_width_chars: int = 0
        
        # Enhanced theme with professional color scheme
        self.theme = {
            'empty': "·",
            'food': "◉",
            'creature': "◈",
            'styles': {
                'title': Style(color="bright_cyan", bold=True),
                'stats': Style(color="bright_white"),
                'grid': Style(color="grey50"),
                'creature': Style(color="bright_green", bold=True),
                'food': Style(color="bright_yellow", bold=True),
                'highlight': Style(color="white", bold=True),
                'border': Style(color="bright_black"),
                'alert': Style(color="bright_red", bold=True),
                'success': Style(color="bright_green"),
                'info': Style(color="bright_blue")
            }
        }
        
        # Enhanced performance metrics
        self.metrics = {
            'start_time': datetime.now(),
            'peak_population': 0,
            'total_food_spawned': 0,
            'births': 0,
            'deaths': 0,
            'total_energy_consumed': 0.0,
            'generations': 0,
            'extinction_events': 0,
            'evolutionary_events': 0
        }
        
        # Track evolutionary history
        self.evolution_history = []
        self.population_history = []
        self.food_history = []

    def spawn_food(self, num: int, tick: int):
        empty_positions = np.nonzero(self.grid == None)
        empty_coords = list(zip(empty_positions[0], empty_positions[1]))
        
        if not empty_coords:
            return
            
        num_to_spawn = min(num, len(empty_coords))
        selected_indices = self.rng.choice(len(empty_coords), size=num_to_spawn, replace=False)
        selected_positions = [empty_coords[i] for i in selected_indices]
        
        for x, y in selected_positions:
            self.grid[x, y] = 'F'
            self.food_spawn_times[(x,y)] = tick
            self.metrics['total_food_spawned'] += 1
            self.logger.log_event("Food spawned", f"({x},{y})", tick)

    def remove_expired_food(self, tick: int, expire_after: int = 20):
        to_remove = [(x,y) for (x,y), t0 in self.food_spawn_times.items() 
                    if (tick - t0) > expire_after]
        
        if to_remove:
            coords = np.array(to_remove)
            self.grid[coords[:,0], coords[:,1]] = None
            for pos in to_remove:
                self.food_spawn_times.pop(pos)
                self.logger.log_event("Food expired", f"({pos[0]},{pos[1]})", tick)

    def add_creature(self, creature: Creature):
        self.creatures.append(creature)

    def step(self, tick: int, food_spawn_interval: int, food_per_spawn: int, food_expire: int, mutation_rate: float):
        if tick % food_spawn_interval == 0:
            self.spawn_food(food_per_spawn, tick)

        if tick % max(1, food_expire) == 0:
            self.remove_expired_food(tick, expire_after=food_expire)

        alive = []
        new_offspring = []
        
        for c in self.creatures:
            if not c.is_alive():
                self.logger.log_event("Creature died", f"Age: {c.age}, Energy: {c.energy:.1f}", tick)
                self.metrics['deaths'] += 1
                self.metrics['total_energy_consumed'] += c.energy
                continue
                
            c.step(self.grid, self.size, self.rng)
            
            if c.can_reproduce():
                child = c.reproduce(self.grid, self.size, self.rng, mutation_rate=mutation_rate)
                if child:
                    new_offspring.append(child)
                    self.metrics['births'] += 1
                    self.metrics['evolutionary_events'] += 1
                    self.logger.log_event("Creature reproduced", 
                        f"Parent ID: {self.logger.creature_ids.get(c, 'unknown')}, " 
                        f"Energy: {c.energy:.1f}, Position: ({c.x},{c.y})", tick)
            
            if c.is_alive():
                alive.append(c)

        for child in new_offspring:
            self.creatures.append(child) 
            alive.append(child)  
            self.logger.record_tick(tick, self.creatures, int(np.sum(self.grid == 'F')))

        self.creatures = alive
        
        # Update metrics
        current_population = len(self.creatures)
        if current_population > self.metrics['peak_population']:
            self.metrics['peak_population'] = current_population
        
        # Record history for visualization
        self.population_history.append(current_population)
        self.food_history.append(int(np.sum(self.grid == 'F')))
        
        # Check for extinction
        if current_population == 0:
            self.metrics['extinction_events'] += 1

        food_count = int(np.sum(self.grid == 'F'))
        self.logger.record_tick(tick, self.creatures, food_count)

    def status_summary(self, tick: int, verbose_every: int = 10):
        if tick % verbose_every == 0:
            food_count = int((self.grid == 'F').sum())
            self.logger.log_event("Status report", f"Creatures: {len(self.creatures)}, Food: {food_count}", tick)
            for c in self.creatures:
                self.logger.log_creature(c, tick)

    def export_history(self, filename="history.csv"):
        self.logger.export_csv(filename)

    def clear_console(self):
        print("\033[2J\033[H", end='')

    def create_layout(self, tick: int) -> Layout:
        """Creates a portfolio-worthy rich layout with perfect scaling"""
        console_width = self.console.width
        console_height = self.console.height

        # Ensure minimum dimensions to prevent cutoff
        min_width = 80
        min_height = 30
        
        if console_width < min_width or console_height < min_height:
            # Fallback to minimal layout for small terminals
            return self._create_minimal_layout(tick)

        # Main layout split with safe margins
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main_content", size=console_height - 11),
            Layout(name="footer", size=6)
        )

        # Main content split with safe ratios
        main_width = max(int(console_width * 0.60), 50)  # Ensure minimum width
        info_width = max(console_width - main_width - 2, 25)  # Ensure minimum info width

        self.layout["main_content"].split_row(
            Layout(name="simulation", size=main_width),
            Layout(name="analytics", size=info_width)
        )

        # Expose panel width for grid sizing/alignment calculations
        self._sim_panel_width_chars = main_width

        # Simulation area split with safe proportions
        grid_height = max(int((console_height - 11) * 0.65), 15)
        stats_height = max(int((console_height - 11) * 0.35), 8)

        self.layout["simulation"].split_column(
            Layout(name="grid_area", size=grid_height),
            Layout(name="simulation_stats", size=stats_height)
        )

        # Analytics area split with safe proportions
        chart_height = max(int((console_height - 11) * 0.35), 8)
        details_height = max(int((console_height - 11) * 0.30), 6)
        metrics_height = max(int((console_height - 11) * 0.35), 8)

        self.layout["analytics"].split_column(
            Layout(name="population_chart", size=chart_height),
            Layout(name="creature_details", size=details_height),
            Layout(name="performance_metrics", size=metrics_height)
        )

        # Update all panels
        self._update_header(tick)
        self._update_grid_area()
        self._update_simulation_stats()
        self._update_population_chart()
        self._update_creature_details()
        self._update_performance_metrics()
        self._update_footer()

        return self.layout

    def _create_minimal_layout(self, tick: int) -> Layout:
        """Create a minimal layout for small terminals"""
        self.layout.split_column(
            Layout(name="header", size=2),
            Layout(name="content", size=20),
            Layout(name="footer", size=4)
        )
        
        self.layout["content"].split_row(
            Layout(name="grid", size=40),
            Layout(name="info", size=30)
        )
        # Best-effort width hint for minimal layout
        self._sim_panel_width_chars = 40
        
        # Simple header
        header_text = Text(f"World Simulation - Tick {tick}", style="bold cyan")
        header_panel = Panel(header_text, box=ROUNDED, border_style="cyan")
        self.layout["header"].update(header_panel)
        
        # Simple grid
        grid_content = self._create_enhanced_grid()
        grid_panel = Panel(grid_content, title="Grid", box=ROUNDED)
        self.layout["grid"].update(grid_panel)
        
        # Simple info
        info_text = Text()
        info_text.append(f"Population: {len(self.creatures)}\n", style="white")
        info_text.append(f"Food: {int(np.sum(self.grid == 'F'))}\n", style="yellow")
        info_text.append(f"Size: {self.size}x{self.size}", style="blue")
        
        info_panel = Panel(info_text, title="Info", box=ROUNDED)
        self.layout["info"].update(info_panel)
        
        # Simple footer
        footer_text = Text("Press Ctrl+C to exit", style="bright_black")
        footer_panel = Panel(footer_text, box=ROUNDED)
        self.layout["footer"].update(footer_panel)
        
        return self.layout

    def _update_header(self, tick: int):
        """Create stunning header with simulation info"""
        runtime = datetime.now() - self.metrics['start_time']
        hours, remainder = divmod(int(runtime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        header_content = Group(
            Align.center(Text("WORLD SIMULATION ENGINE", style="bold bright_cyan")),
            Align.center(Text(f"Tick {tick:06d} | Runtime: {hours:02d}:{minutes:02d}:{seconds:02d} | Population: {len(self.creatures)}", style="bright_white")),
            Rule(style="bright_black")
        )
        
        header_panel = Panel(
            header_content,
            box=DOUBLE,
            border_style="bright_cyan",
            padding=(0, 2)
        )
        
        self.layout["header"].update(header_panel)

    def _update_grid_area(self):
        """Create the main simulation grid with professional styling"""
        grid_content = self._create_enhanced_grid()
        
        grid_panel = Panel(
            grid_content,
            title="[bright_white]Simulation Grid",
            border_style="bright_black",
            box=HEAVY,
            padding=(0, 1)
        )
        
        self.layout["grid_area"].update(grid_panel)

    def _create_enhanced_grid(self) -> Text:
        """Render a perfectly aligned, monospaced grid with headers.

        Uses fixed-width cells and ASCII/block characters to avoid ambiguous
        Unicode widths that cause misalignment in some terminals.
        """
        grid_text = Text()

        # Display up to a sensible cap but prefer full world if small
        max_display_size = self.size

        # Fixed-size cell design for rock-solid alignment (char + space)
        # This avoids reliance on panel width calculations which vary by theme/padding.
        row_label_width = max(2, len(str(max_display_size - 1)))
        cell_gap = " "  # one space between columns

        # Characters chosen for reliable monospace width (ASCII-safe)
        empty_char = "."
        food_char = "*"
        creature_char = "@"

        # Precompute creature positions for O(1) lookups
        creature_positions = {(c.x, c.y) for c in self.creatures}

        # Column headers (two-line: tens then ones) using fixed column width of 2 (digit+space)
        grid_text.append(" " * row_label_width + " ", style="bright_black")
        for j in range(max_display_size):
            tens = str(j // 10) if j >= 10 else " "
            grid_text.append(tens + cell_gap, style="bright_blue")
        grid_text.append("\n")
        grid_text.append(" " * row_label_width + " ", style="bright_black")
        for j in range(max_display_size):
            ones = str(j % 10)
            grid_text.append(ones + cell_gap, style="bright_blue")
        grid_text.append("\n")

        # Rows
        for i in range(max_display_size):
            # Row label
            grid_text.append(f"{i:>{row_label_width}} ", style="bright_blue")
            # Cells (fixed width: 2)
            for j in range(max_display_size):
                if (i, j) in creature_positions:
                    grid_text.append(creature_char + cell_gap, style=self.theme['styles']['creature'])
                elif self.grid[i, j] == 'F':
                    grid_text.append(food_char + cell_gap, style=self.theme['styles']['food'])
                else:
                    grid_text.append(empty_char + cell_gap, style=self.theme['styles']['grid'])
            grid_text.append("\n")

        if max_display_size < self.size:
            grid_text.append(
                f"\nGrid truncated: {max_display_size}x{max_display_size} of {self.size}x{self.size}",
                style="bright_black",
            )

        return grid_text

    def _update_simulation_stats(self):
        """Create simulation statistics panel"""
        stats_content = self._create_simulation_stats()
        
        stats_panel = Panel(
            stats_content,
            title="[bright_white]Live Statistics",
            border_style="bright_black",
            box=ROUNDED,
            padding=(0, 1)
        )
        
        self.layout["simulation_stats"].update(stats_panel)

    def _create_simulation_stats(self) -> Group:
        """Create enhanced simulation statistics"""
        # Population overview
        pop_table = Table(show_header=False, box=ROUNDED, show_lines=False, padding=0)
        pop_table.add_column("Metric", style="bright_white")
        pop_table.add_column("Value", style="bright_cyan")
        
        current_pop = len(self.creatures)
        max_pop = max(self.metrics['peak_population'], current_pop, 1)
        population_bar = ProgressBar(total=max_pop, completed=current_pop, width=20)
        
        pop_table.add_row("Population", f"{current_pop}")
        pop_table.add_row("", population_bar)
        pop_table.add_row("Food Available", f"{int(np.sum(self.grid == 'F'))}")
        pop_table.add_row("Grid Size", f"{self.size}x{self.size}")
        
        # Top creatures
        if self.creatures:
            pop_table.add_section()
            pop_table.add_row("Top Creatures", "", style="bright_yellow")
            for idx, c in enumerate(sorted(self.creatures, key=lambda x: x.energy, reverse=True)[:3]):
                energy_pct = min(100, int((c.energy / 150) * 100))
                energy_bar = ProgressBar(total=100, completed=energy_pct, width=15)
                pop_table.add_row(
                    f"#{self.logger.creature_ids.get(c, '??')}",
                    f"E:{c.energy:>5.1f} A:{c.age:>3d}"
                )
                pop_table.add_row("", energy_bar)
        
        return Group(pop_table)

    def _update_population_chart(self):
        """Create population trend visualization"""
        chart_content = self._create_population_chart()
        
        chart_panel = Panel(
            chart_content,
            title="[bright_white]Population Trends",
            border_style="bright_black",
            box=ROUNDED,
            padding=(0, 1)
        )
        
        self.layout["population_chart"].update(chart_panel)

    def _create_population_chart(self) -> Text:
        """Create ASCII population chart with proper sizing"""
        chart_text = Text()
        
        if len(self.population_history) < 2:
            chart_text.append("Collecting data...", style="bright_black")
            return chart_text
        
        # Create compact ASCII chart
        max_pop = max(self.population_history) if self.population_history else 1
        chart_height = 6  # Reduced height to prevent cutoff
        max_data_points = 15  # Limit data points to prevent horizontal cutoff
        
        for i in range(chart_height, 0, -1):
            threshold = (i / chart_height) * max_pop
            line = ""
            # Use only recent data points to prevent cutoff
            recent_data = self.population_history[-max_data_points:]
            for pop in recent_data:
                if pop >= threshold:
                    line += "█"
                else:
                    line += " "
            chart_text.append(f"{int(threshold):2d} ", style="bright_blue")
            chart_text.append(line, style="bright_green")
            chart_text.append("\n")
        
        # Compact axis
        chart_text.append("   ", style="bright_blue")
        chart_text.append("─" * len(recent_data), style="bright_black")
        chart_text.append("\n")
        chart_text.append("   ", style="bright_blue")
        chart_text.append("Time", style="bright_white")
        
        return chart_text

    def _update_creature_details(self):
        """Create detailed creature information panel"""
        details_content = self._create_creature_details()
        
        details_panel = Panel(
            details_content,
            title="[bright_white]Creature Details",
            border_style="bright_black",
            box=ROUNDED,
            padding=(0, 1)
        )
        
        self.layout["creature_details"].update(details_panel)

    def _create_creature_details(self) -> Group:
        """Create enhanced creature details with compact formatting"""
        if not self.creatures:
            return Group(Text("No creatures alive", style="bright_red"))
        
        # Find most interesting creatures (limit to 2 for small panels)
        top_creatures = sorted(self.creatures, key=lambda x: x.energy, reverse=True)[:2]
        
        details_group = []
        for i, c in enumerate(top_creatures):
            creature_text = Text()
            creature_text.append(f"#{self.logger.creature_ids.get(c, '??')} ", style="bright_cyan")
            creature_text.append(f"({c.x},{c.y})", style="bright_white")
            creature_text.append("\n")
            creature_text.append(f"E:{c.energy:.1f} A:{c.age}", style="bright_green")
            creature_text.append("\n")
            creature_text.append(f"V:{c.traits.vision} S:{c.traits.speed}", style="bright_blue")
            creature_text.append(f" M:{c.traits.metabolism:.2f}", style="bright_red")
            
            if i < len(top_creatures) - 1:
                creature_text.append("\n" + "─" * 15 + "\n")  # Shorter separator
            
            details_group.append(creature_text)
        
        return Group(*details_group)

    def _update_performance_metrics(self):
        """Create performance metrics panel"""
        metrics_content = self._create_performance_metrics()
        
        metrics_panel = Panel(
            metrics_content,
            title="[bright_white]Performance Metrics",
            border_style="bright_black",
            box=ROUNDED,
            padding=(0, 1)
        )
        
        self.layout["performance_metrics"].update(metrics_panel)

    def _create_performance_metrics(self) -> Table:
        """Create enhanced performance metrics with compact formatting"""
        table = Table(show_header=False, box=ROUNDED, show_lines=False, padding=0)
        table.add_column("Metric", style="bright_white")
        table.add_column("Value", style="bright_cyan")
        
        # Calculate rates
        runtime = datetime.now() - self.metrics['start_time']
        runtime_minutes = runtime.total_seconds() / 60
        
        birth_rate = self.metrics['births'] / max(1, runtime_minutes)
        death_rate = self.metrics['deaths'] / max(1, runtime_minutes)
        
        # Compact metric names to prevent cutoff
        table.add_row("Birth Rate", f"{birth_rate:.1f}/min")
        table.add_row("Death Rate", f"{death_rate:.1f}/min")
        table.add_row("Total Born", str(self.metrics['births']))
        table.add_row("Total Died", str(self.metrics['deaths']))
        table.add_row("Peak Pop", str(self.metrics['peak_population']))  # Shortened
        table.add_row("Evol Events", str(self.metrics['evolutionary_events']))  # Shortened
        table.add_row("Food Count", str(self.metrics['total_food_spawned']))  # Shortened
        table.add_row("Energy Used", f"{self.metrics['total_energy_consumed']:.0f}")  # Rounded
        
        return table

    def _update_footer(self):
        """Create enhanced footer with recent events"""
        footer_content = self._create_enhanced_footer()
        
        footer_panel = Panel(
            footer_content,
            title="[bright_white]Recent Events",
            border_style="bright_black",
            box=ROUNDED,
            padding=(0, 1)
        )
        
        self.layout["footer"].update(footer_panel)

    def _create_enhanced_footer(self) -> Text:
        """Create enhanced footer content"""
        events = self.logger.get_recent_events(4)
        event_text = Text()
        
        if not events:
            event_text.append("No events recorded yet...", style="bright_black")
            return event_text
        
        for event in events:
            # Color code different event types
            if "died" in event['event'].lower():
                event_text.append("● ", style="bright_red")
            elif "reproduced" in event['event'].lower():
                event_text.append("● ", style="bright_green")
            elif "food" in event['event'].lower():
                event_text.append("● ", style="bright_yellow")
            else:
                event_text.append("● ", style="bright_blue")
            
            event_text.append(f"{event['event']}: ", style="bright_white")
            event_text.append(f"{event['details']}", style="bright_cyan")
            event_text.append("\n")
        
        return event_text

    def display_world(self, tick: int):
        """Legacy method for backward compatibility"""
        layout = self.create_layout(tick)
        self.console.print("\x1b[2J\x1b[H", end="")
        self.console.print(layout)
