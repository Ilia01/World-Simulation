from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..core import World
from ..core.entities import Creature
from ..logging import WorldLogger


def _default_config() -> Dict[str, Any]:
    return {
        "world_size": 20,
        "initial_creatures": 8,
        "max_ticks": 10_000,
        "food_spawn_interval": 3,
        "food_per_spawn": 5,
        "food_expire": 20,
        "mutation_rate": 0.08,
        "refresh_rate": 3,
        "seed": 42,
    }


def _ensure_state() -> None:
    if "config" not in st.session_state:
        st.session_state.config = _default_config()
    if "world" not in st.session_state:
        st.session_state.world: Optional[World] = None
    if "logger" not in st.session_state:
        st.session_state.logger: Optional[WorldLogger] = None
    if "tick" not in st.session_state:
        st.session_state.tick = 0
    if "running" not in st.session_state:
        st.session_state.running = False


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
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


def _lineage_color(lineage: Optional[int]) -> np.ndarray:
    if lineage is None:
        return np.array([56, 255, 146], dtype=np.uint8)
    hue = (hash(lineage) % 360) / 360.0
    r, g, b = _hsv_to_rgb(hue, 0.75, 1.0)
    return np.array([r, g, b], dtype=np.uint8)


def _render_grid_image(
    world: World, cell_px: int = 16, show_gridlines: bool = True, lineage_colors: bool = False
) -> np.ndarray:
    size = world.size
    color_empty = np.array([18, 18, 22], dtype=np.uint8)
    color_food = np.array([255, 204, 0], dtype=np.uint8)
    default_creature = np.array([56, 255, 146], dtype=np.uint8)
    img = np.tile(color_empty, (size, size, 1))
    food_mask = world.grid == "F"
    img[food_mask] = color_food
    for c in world.creatures:
        if 0 <= c.x < size and 0 <= c.y < size:
            base_color = _lineage_color(c.lineage) if lineage_colors else default_creature
            energy_scale = min(1.0, max(0.35, c.energy / 30.0))
            img[c.x, c.y] = (base_color.astype(np.float32) * energy_scale).astype(np.uint8)
    img = np.rot90(img, k=1)
    if cell_px > 1:
        img = img.repeat(cell_px, axis=0).repeat(cell_px, axis=1)
    if show_gridlines and cell_px >= 6:
        grid_color = np.array([32, 36, 44], dtype=np.uint8)
        h, w, _ = img.shape
        for y in range(0, h, cell_px):
            img[y : y + 1, :, :] = grid_color
        for x in range(0, w, cell_px):
            img[:, x : x + 1, :] = grid_color
    return img


def _history_dataframe(world: World) -> pd.DataFrame:
    if not world.logger.history:
        return pd.DataFrame({"tick": [], "creature_count": [], "food_count": [], "avg_energy": [], "max_age": []})
    return pd.DataFrame(world.logger.history)


def _make_plotly_figure(world: World, cfg: Dict[str, Any]) -> go.Figure:
    # Base image layer
    img = _render_grid_image(
        world,
        cell_px=int(cfg.get("cell_px", 18)),
        show_gridlines=bool(cfg.get("show_gridlines", True)),
        lineage_colors=bool(cfg.get("lineage_colors", False)),
    )
    fig = px.imshow(img, binary_string=False, origin="upper")
    fig.update_traces(hoverinfo="skip")

    # Overlay creature markers for crisp, elegant look
    if bool(cfg.get("overlay_points", True)) and world.creatures:
        xs = []
        ys = []
        colors = []
        sizes = []
        texts = []
        for c in world.creatures:
            # Note rotation used in image; y is horizontal axis after rot90
            xs.append(c.y)
            ys.append(world.size - 1 - c.x)
            sizes.append(max(6, min(14, int(6 + (c.energy / 30.0) * 10))))
            texts.append(
                f"ID {world.logger.creature_ids.get(c,'?')} • pos=({c.x},{c.y})\n"
                f"E={c.energy:.1f} A={c.age} V={c.traits.vision} S={c.traits.speed} M={c.traits.metabolism:.2f}"
            )

            mode = str(cfg.get("point_color_mode", "Lineage"))
            if mode == "Energy":
                # Map energy to green intensity
                e = max(0.0, min(1.0, c.energy / 30.0))
                r, g, b = int(56 * e), int(255 * e), int(146 * e)
                colors.append(f"rgb({r},{g},{b})")
            elif mode == "Age":
                t = max(0.0, min(1.0, c.age / 100.0))
                r = int(255 * t)
                b = int(255 * (1 - t))
                colors.append(f"rgb({r},128,{b})")
            elif mode == "Solid":
                colors.append("rgb(56,255,146)")
            else:  # Lineage
                rgb = _lineage_color(c.lineage)
                colors.append(f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})")

        fig.add_trace(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(color=colors, size=sizes, line=dict(width=0)),
                hovertext=texts,
                hoverinfo="text",
                name="Creatures",
            )
        )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, showgrid=False, zeroline=False, constrain="domain"),
        yaxis=dict(visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        dragmode=False,
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True, autorange="reversed")
    return fig


def _controls_sidebar() -> Dict[str, Any]:
    st.sidebar.title("Controls")
    st.sidebar.caption("Tune parameters and control execution")

    cfg = st.session_state.config
    world_size = st.sidebar.slider("World Size", 10, 50, cfg["world_size"], 1)
    initial_creatures = st.sidebar.slider("Initial Creatures", 1, 30, cfg["initial_creatures"], 1)
    mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.20, float(cfg["mutation_rate"]), 0.01)
    refresh_rate = st.sidebar.slider("Refresh Rate (Hz)", 1, 20, int(cfg["refresh_rate"]), 1)
    food_spawn_interval = st.sidebar.slider("Food Spawn Interval", 1, 10, cfg["food_spawn_interval"], 1)
    food_per_spawn = st.sidebar.slider("Food per Spawn", 1, 20, cfg["food_per_spawn"], 1)
    food_expire = st.sidebar.slider("Food Expire (ticks)", 5, 50, cfg["food_expire"], 1)
    seed = st.sidebar.number_input("Seed", value=int(cfg["seed"]), step=1)

    st.sidebar.markdown("---")
    st.sidebar.caption("Grid appearance")
    cell_px = st.sidebar.slider("Cell size (px)", 6, 40, int(cfg.get("cell_px", 18)), 1)
    show_gridlines = st.sidebar.toggle("Show gridlines", value=bool(cfg.get("show_gridlines", True)))
    lineage_colors = st.sidebar.toggle(
        "Use lineage colors for creatures (image)", value=bool(cfg.get("lineage_colors", False))
    )
    render_plotly = st.sidebar.toggle(
        "Use Plotly renderer (reduces flicker)", value=bool(cfg.get("render_plotly", True))
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Overlay settings")
    overlay_points = st.sidebar.toggle("Overlay creature markers", value=bool(cfg.get("overlay_points", True)))
    point_color_mode = st.sidebar.selectbox(
        "Marker color by",
        options=["Lineage", "Energy", "Age", "Solid"],
        index=["Lineage", "Energy", "Age", "Solid"].index(str(cfg.get("point_color_mode", "Lineage"))),
    )

    st.session_state.config.update(
        {
            "world_size": int(world_size),
            "initial_creatures": int(initial_creatures),
            "mutation_rate": float(mutation_rate),
            "refresh_rate": int(refresh_rate),
            "food_spawn_interval": int(food_spawn_interval),
            "food_per_spawn": int(food_per_spawn),
            "food_expire": int(food_expire),
            "seed": int(seed),
            "cell_px": int(cell_px),
            "show_gridlines": bool(show_gridlines),
            "lineage_colors": bool(lineage_colors),
            "render_plotly": bool(render_plotly),
            "overlay_points": bool(overlay_points),
            "point_color_mode": str(point_color_mode),
        }
    )

    c1, c2, c3, c4 = st.sidebar.columns(4)
    if c1.button("Init", help="Initialize / Reset world"):
        _initialize_world(st.session_state.config)
        st.session_state.running = False
    if c2.button("Step", help="Advance one tick"):
        _step_world_once()
    if c3.button("Run" if not st.session_state.running else "Pause", type="primary"):
        st.session_state.running = not st.session_state.running
    if c4.button("Export", help="Export CSV logs"):
        if st.session_state.logger:
            st.session_state.logger.events_file.flush()
            st.session_state.logger.creatures_file.flush()
            st.success("Exported events.csv and creatures.csv")

    with st.sidebar.expander("Downloads"):
        try:
            with open("events.csv", "rb") as f:
                st.download_button("Download events.csv", f, file_name="events.csv")
        except Exception:
            st.caption("events.csv not available yet")
        try:
            with open("creatures.csv", "rb") as f:
                st.download_button("Download creatures.csv", f, file_name="creatures.csv")
        except Exception:
            st.caption("creatures.csv not available yet")

    return st.session_state.config


def _initialize_world(config: Dict[str, Any]) -> None:
    prev_logger: Optional[WorldLogger] = st.session_state.get("logger")
    if prev_logger is not None:
        try:
            prev_logger.close()
        except Exception:
            pass
    logger = WorldLogger()
    world = World(size=config["world_size"], rng_seed=config["seed"], logger=logger)
    rng = world.rng
    for i in range(config["initial_creatures"]):
        x = int(rng.integers(0, world.size))
        y = int(rng.integers(0, world.size))
        creature = Creature(x, y, energy=15.0)
        creature.lineage = i
        world.add_creature(creature)
    logger.log_event("World initialized", f"Size: {world.size}x{world.size}")
    logger.log_event("Creatures created", f"Count: {config['initial_creatures']}")
    st.session_state.world = world
    st.session_state.logger = logger
    st.session_state.tick = 0


def _step_world_once() -> None:
    world: Optional[World] = st.session_state.world
    if not world:
        _initialize_world(st.session_state.config)
        world = st.session_state.world
    tick = st.session_state.tick
    cfg = st.session_state.config
    world.step(
        tick,
        food_spawn_interval=cfg["food_spawn_interval"],
        food_per_spawn=cfg["food_per_spawn"],
        food_expire=cfg["food_expire"],
        mutation_rate=cfg["mutation_rate"],
    )
    st.session_state.tick = tick + 1


def _hero_header() -> None:
    left, right = st.columns([0.65, 0.35])
    with left:
        st.markdown(
            """
            ### 🌍 World Simulation – Portfolio Dashboard
            A modern, interactive artificial life simulation featuring evolutionary dynamics and real-time analytics.
            """
        )
    with right:
        if st.session_state.world:
            w = st.session_state.world
            metrics = w.metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Population", len(w.creatures))
            c2.metric("Peak", metrics.get("peak_population", 0))
            c3.metric("Births", metrics.get("births", 0))
            # Add a compact KPI row
            c4, c5, c6 = st.columns(3)
            c4.metric("Deaths", metrics.get("deaths", 0))
            c5.metric("Food Spawned", metrics.get("total_food_spawned", 0))
            c6.metric("Evol Events", metrics.get("evolutionary_events", 0))


def _main_content(config: Dict[str, Any]) -> None:
    world: Optional[World] = st.session_state.world
    grid_col, info_col = st.columns([0.6, 0.4])
    with grid_col:
        st.subheader("Simulation Grid")
        if world is None:
            st.info("Click Init to create the world.")
        else:
            cfg = st.session_state.config
            if bool(cfg.get("render_plotly", True)):
                fig = _make_plotly_figure(world, cfg)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "staticPlot": True})
            else:
                img = _render_grid_image(
                    world,
                    cell_px=int(cfg.get("cell_px", 18)),
                    show_gridlines=bool(cfg.get("show_gridlines", True)),
                    lineage_colors=bool(cfg.get("lineage_colors", False)),
                )
                st.image(
                    img, caption=f"Tick {st.session_state.tick}", use_container_width=True, channels="RGB", output_format="PNG"
                )
            st.caption("Yellow: Food • Green/Palette: Creatures • Dark: Empty")

    with info_col:
        st.subheader("Live Analytics")
        if world is None:
            st.caption("No data yet.")
        else:
            df = _history_dataframe(world)
            if not df.empty:
                chart_cols = st.columns(2)
                with chart_cols[0]:
                    st.line_chart(df.set_index("tick")["creature_count"], height=180)
                    st.caption("Population over time")
                with chart_cols[1]:
                    st.line_chart(df.set_index("tick")["food_count"], height=180)
                    st.caption("Food availability over time")

                extra_cols = st.columns(2)
                with extra_cols[0]:
                    st.area_chart(df.set_index("tick")["avg_energy"], height=150)
                    st.caption("Average energy")
                with extra_cols[1]:
                    st.bar_chart(df.set_index("tick")["max_age"], height=150)
                    st.caption("Max age per tick")
                with st.expander("Raw history data"):
                    st.dataframe(df.tail(100), use_container_width=True, hide_index=True)
            else:
                st.caption("Collecting data...")

        st.divider()
        st.subheader("Recent Events")
        if world and st.session_state.logger:
            events = st.session_state.logger.get_recent_events(num=8)
            for e in events[::-1]:
                st.write(
                    f"[{e.get('timestamp','')}] Tick {e.get('tick','-')}: {e.get('event','')} — {e.get('details','')}"
                )
        else:
            st.caption("No events yet.")


def run() -> None:
    st.set_page_config(page_title="World Simulation – Portfolio Dashboard", page_icon="🌍", layout="wide")
    _ensure_state()
    config = _controls_sidebar()
    _hero_header()
    _main_content(config)
    if st.session_state.running:
        _step_world_once()
        time.sleep(max(0.01, 1.0 / max(1, config["refresh_rate"])))
        st.rerun()


