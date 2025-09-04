import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def analyze_simulation_data(
    creatures_file: str = "creatures.csv", events_file: str = "events.csv", history_file: str = "history.csv"
):
    # Load CSV files
    creatures_df = pd.read_csv(
        creatures_file,
        dtype={
            "tick": np.int32,
            "creature_id": np.int32,
            "x": np.int16,
            "y": np.int16,
            "energy": np.float32,
            "age": np.int32,
            "vision": np.int16,
            "speed": np.int16,
            "metabolism": np.float32,
        },
    )
    events_df = pd.read_csv(events_file)
    history_df = pd.read_csv(history_file)

    # === SUMMARY ===
    unique_creatures = creatures_df["creature_id"].nunique()
    population_per_tick = creatures_df.groupby("tick")["creature_id"].nunique()
    avg_energy_per_tick = creatures_df.groupby("tick")["energy"].mean()
    max_age_per_tick = creatures_df.groupby("tick")["age"].max()

    # === PLOTTING ===
    plt.style.use("ggplot")
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    fig.suptitle("World Simulation – Analytics", fontsize=16)

    population_per_tick.plot(ax=axes[0, 0], title="Population Over Time", color="tab:blue")
    axes[0, 0].set_xlabel("Tick")
    axes[0, 0].set_ylabel("Creatures")

    avg_energy_per_tick.plot(ax=axes[0, 1], title="Average Energy Over Time", color="tab:green")
    axes[0, 1].set_xlabel("Tick")
    axes[0, 1].set_ylabel("Energy")

    max_age_per_tick.plot(ax=axes[1, 0], title="Max Age Over Time", color="tab:orange")
    axes[1, 0].set_xlabel("Tick")
    axes[1, 0].set_ylabel("Max Age")

    hist_ax = axes[1, 1]
    last_tick = creatures_df["tick"].max()
    ages_last_tick = creatures_df[creatures_df["tick"] == last_tick]["age"]
    hist_ax.hist(ages_last_tick, bins=20, color="tab:gray", edgecolor="black")
    hist_ax.set_title(f"Age Distribution at Tick {last_tick}")
    hist_ax.set_xlabel("Age")
    hist_ax.set_ylabel("Count")

    events_per_tick = events_df["event"].groupby(events_df["tick"]).count()
    events_per_tick.plot(ax=axes[2, 0], kind="bar", color="tab:purple")
    axes[2, 0].set_title("Events Per Tick")
    axes[2, 0].set_xlabel("Tick")
    axes[2, 0].set_ylabel("Events")

    # Cross-check history
    axes[2, 1].plot(history_df["tick"], history_df["creature_count"], label="Population", color="tab:blue")
    axes[2, 1].plot(history_df["tick"], history_df["food_count"], label="Food", color="tab:green")
    axes[2, 1].set_title("History Snapshot")
    axes[2, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    analyze_simulation_data()
