# 🌍 World Simulation

<p align="left">
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green"></a>
</p>

## 🔎 What is this?
Artificial life simulation with evolving agents, real-time visualization, and production-like structure. Built with clean architecture, typed Python, and both CLI and web UIs.

## 💡 Why it matters
- Systems design: agents, events, logging, UI separation
- Production signals: package layout, dependency management, data export, analytics
- Practical: fast local runs; easy to extend with new behaviors/traits

## 🧱 Technical highlights
- `world_sim/core`: pure simulation engine (no UI coupling)
- `world_sim/ui/cli.py` and `app.py`: terminal and Streamlit frontends
- Structured CSV logging via `world_sim/logging/logger.py`
- Deterministic runs with RNG seeding; configurable world parameters

## ⚡ Run in 2 minutes
```bash
git clone https://github.com/Ilia01/World_Simulation.git
cd World_Simulation
pip install -r requirements.txt

# Terminal UI
python main.py

# Web UI
streamlit run app.py
```

## 📸 Screenshots / Demo
<p align="center">
  <img src="docs/assets/web_demo.png" alt="Web dashboard" width="48%" />
  <img src="docs/assets/terminal_demo.png" alt="Terminal UI" width="48%" />
</p>
<p align="center">
  <i>If images are missing, see live UI by running the commands above.</i>
</p>

## 🧰 Tech Stack
<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" />
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" />
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img alt="Rich" src="https://img.shields.io/badge/Rich-1F2328?logo=python&logoColor=white" />
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white" />
</p>

## More docs
See `docs/README.md` for full architecture, algorithms, and customization.

## License
MIT — see `LICENSE`.

## 🗺️ Roadmap
- [ ] Add lineage heatmap and creature detail panel in web UI
- [ ] Add unit tests for behaviors and world mechanics
- [ ] Optional Dockerfile for one-command run
- [ ] GitHub Actions CI (lint + tests)
