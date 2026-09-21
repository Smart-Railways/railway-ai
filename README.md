# 🚆 Railway AI — Intelligent Block Planning Engine

> **Smart, coordinated maintenance block scheduling for Indian Railways' high-density corridors.**  
> Focused on the **New Delhi – Mumbai Golden Quadrilateral route (1,384 km)**.

---

## 📖 The Big Picture: Why We Built This

Anyone who has traveled on Indian Railways knows that punctuality and safety are a delicate balancing act. Every single day, thousands of kilometers of track, overhead electric traction wires, and signal systems take a heavy pounding from passenger expresses and heavy freight trains.

To keep everything running safely, railway maintenance crews need time on the tracks. In railway terms, this is called a **"maintenance block"** — a window of time (usually 2 to 6 hours) where a specific stretch of track is closed to train traffic so engineers can replace rails, inspect signals, or adjust high-voltage wires.

### The Everyday Dilemma
1. **Safety vs. Punctuality**: If you close a track during peak hours, you delay dozens of passenger trains, creating cascading delays that ripple hundreds of kilometers down the line. But if you postpone maintenance too long, a rail fracture or signal failure halts everything for hours.
2. **The Silo Problem**: Track engineers (Engineering/Civil), signal technicians (Signal & Telecom), and electric wire crews (Traction/TRD) belong to different departments. Historically, each team requested their own separate blocks on different days. This meant the same section of track was shut down multiple times for separate repairs.
3. **Information Overload**: Section controllers often have to make high-stakes scheduling decisions over phone calls and paper logs, with limited real-time visibility into which track sections are under the heaviest delay pressure right now.

**Railway AI is built to solve this.** It acts as an intelligent co-pilot for section controllers and railway engineers, finding optimal, conflict-free maintenance slots that keep tracks safe while keeping train delays to an absolute minimum.

---

## ⚡ What Does the System Actually Do?

Railway AI combines predictive track health analytics with constraint optimization and live corridor train tracking. Here is how it works in plain terms:

### 1. Predicts Trouble Before It Happens
Instead of waiting for an asset to fail on the tracks, the machine learning models look at:
- Asset age and installation vintage
- Track condition score and degradation history
- Track route criticality (mainline vs. loop line classification)
- Section usage intensity (traffic loading ratio)
- Historical failure counts and cumulative downtime hours
- Days elapsed since the last recorded failure

It outputs a calibrated **failure probability** for the next 30 days and feeds directly into multi-factor maintenance prioritization.

### 2. Listens to Real Train Traffic & Delay Pressure
The system connects directly with live train telemetry along the corridor (capturing flagship trains like the *12002 Bhopal Shatabdi*, *12301 Howrah Rajdhani*, *12904 Golden Temple Mail*, and *20164 Vande Bharat*). It monitors real-time running delays across key sections and assigns dynamic **operational pressure scores** — so the system avoids scheduling blocks when a section is already struggling with congestion or severe delays.

### 3. Solves the Multi-Window Block Puzzle (CP-SAT Optimization)
Scheduling 15 or 20 maintenance jobs across a busy corridor is a complex mathematical puzzle. The system uses Google OR-Tools constraint satisfaction (CP-SAT) to generate schedules that guarantee:
- **No track conflicts**: Two jobs never block the same section at overlapping times.
- **Crew and equipment limits**: Maintenance gangs aren't double-booked beyond available manpower.
- **Window fit**: Heavy mechanized tampers get the continuous 4-hour slots they need, while quick inspections fit into shorter 2-hour gaps.

### 4. Bundles Multi-Department Work ("One Closure, Three Jobs Done")
When track engineers shut down a section between Mathura and Agra for rail renewal, the system proactively scans for pending signal checkups and overhead traction inspections in that exact section. It bundles them into the same block window, applying a **coordination bonus** that cuts unnecessary track closures by up to **40%**.

### 5. Multi-Horizon Planning: Daily to Monthly
Maintenance isn't just planned day-to-day. The engine supports:
- **Daily Schedules**: Immediate 6-hour and 24-hour tactical block assignments.
- **7-Day Weekly Plans**: Coordinated multi-day maintenance programs.
- **30-Day Monthly Lookahead**: Strategic long-term maintenance roadmaps.
- **Dynamic Rescheduling**: If a sudden delay or emergency occurs on the corridor, the engine locks already-completed work and dynamically re-plans the remaining jobs on the fly.

### 6. Transparent & Human-in-the-Loop (Top-3 Recommendations)
We do not believe in black-box or autonomous AI for mission-critical railways. CP-SAT acts strictly as a decision-support recommendation engine:
- For each task, CP-SAT produces **at most 3 ranked feasible block window recommendations** (`rank 1`, `rank 2`, `rank 3`).
- Every recommendation includes plain-English reason tags (e.g., `OPTIMAL_PRIORITY_FIT`, `LOWEST_OPERATIONAL_DELAY`, `CROSS_DEPARTMENT_COORDINATION`).
- CP-SAT produces ranked feasible recommendations; it does **not** book the final maintenance slot.
- The human operator/controller chooses the final maintenance slot.

---

## 🗺️ Corridor Coverage

The project focuses on the **New Delhi to Mumbai corridor** (1,384 km), one of the busiest passenger and freight arteries in Asia:

```text
[NDLS] New Delhi
   │
   ▼
[MTJ]  Mathura Junction
   │
   ▼
[AGC]  Agra Cantt
   │
   ▼
[GWL]  Gwalior
   │
   ▼
[VGLJ] Virangana Lakshmibai Jhansi (formerly JHS)
   │
   ▼
[BINA] Bina Junction
   │
   ▼
[BPL]  Bhopal Junction
   │
   ▼
[BRC]  Vadodara Junction (formerly VAD)
   │
   ▼
[ST]   Surat (formerly SRT)
   │
   ▼
[MMCT] Mumbai Central (formerly MUM)
```

The system automatically resolves historical and modern Indian Railways station code aliases (such as `JHS` ➔ `VGLJ`, `VAD` ➔ `BRC`, `SRT` ➔ `ST`, and `MUM` ➔ `MMCT`) using verified corridor mapping evidence.

---

## 📂 Repository Layout

Here is how the project is organized:

```text
railway-ai/
├── src/                               # Core engine modules
│   ├── decision/                      # Multi-factor priority & decision scoring engine
│   │   └── maintenance_decision_engine.py
│   ├── optimization/                  # CP-SAT block optimizer & multi-horizon planners
│   │   ├── block_optimizer.py         # 6-hour discrete time slot solver
│   │   └── multi_horizon_planner.py   # Weekly, monthly, rolling & dynamic scheduler
│   ├── models/                        # ML inference wrappers & risk scoring
│   │   └── failure_predictor.py       # Calibrated risk & threshold management
│   ├── features/                      # Operational telemetry & delay pressure builder
│   │   ├── real_operational_features.py
│   │   ├── section_pressure_builder.py
│   │   ├── operational_context_enricher.py
│   │   └── railkit_feature_bridge.py
│   ├── data/                          # Telemetry extraction & station sequence tools
│   │   ├── railkit_client.py
│   │   └── section_evidence_builder.py
│   ├── services/                      # High-level engine facade & service interfaces
│   │   ├── ml_engine.py               # Unified RailwayMLEngine entrypoint
│   │   ├── planning_service.py        # Planning orchestration service
│   │   └── real_corridor_planning.py  # Live corridor integration service
│   └── evaluation/                    # Benchmarking against uncoordinated baselines
│       └── benchmark_validator.py
│
├── config/                            # Machine-readable configurations & manifests
│   ├── corridor_master.json           # Corridor stations, distances & speed limits
│   ├── railkit_section_mapping.json   # Station sequence & section definitions
│   ├── pipeline_manifest.json         # Phase-by-phase lifecycle audit
│   ├── data_source_contract.json      # Schema definitions for data ingestion
│   └── ml_contract.json               # Input/output contracts for external services
│
├── data/
│   ├── raw_real/railkit/              # Real RailKit telemetry captures (JSON)
│   └── processed_real/                # Cleaned corridor section mapping evidence (CSV)
│
├── models/                            # Organized model artifacts
│   ├── production/                    # Active production model (Candidate V3)
│   │   └── calibrated_xgboost.pkl
│   ├── legacy/                        # Archived legacy model (V2 baseline)
│   │   └── calibrated_xgboost_v2_legacy.pkl
│   ├── candidate/                     # Evaluated candidate models & manifests
│   │   └── candidate_v3/
│   └── experimental/                  # Deep learning, survival & research checkpoints
│
├── scripts/                           # Reusable operational & evaluation scripts
│   ├── training/                      # Model training pipelines
│   ├── evaluation/                    # Independent audit & candidate evaluation
│   ├── validation/                    # Feature leakage & inference smoke tests
│   ├── data/                          # Target & dataset generation pipelines
│   └── maintenance/                   # Rollback runbooks & verification utilities
│
├── tests/                             # 160 automated pytest unit, integration & contract tests
│   ├── model/                         # Risk predictor, candidate v3 & benchmark validation
│   ├── decision/                      # 5-factor priority formula & regression tests
│   ├── scheduling/                    # CP-SAT Top-3 recommendations & multi-horizon tests
│   ├── lifecycle/                     # Real-time state machine & human confirmation tests
│   ├── contracts/                     # Backend schema & 4-category contract tests
│   ├── data/                          # Corridor evidence & telemetry tests
│   └── integration/                   # Unified RailwayMLEngine end-to-end tests
│
├── schemas/                           # Machine-readable integration schemas
│   └── ml_response.schema.json        # Frozen 4-category backend response schema
│
├── docs/                              # Project documentation & audit reports
│   ├── architecture/                  # Architectural references & design
│   ├── model/                         # ML status & model improvement reports
│   ├── integration/                   # Backend contract & defense guides
│   └── audits/                        # Promotion readiness, final audits & verification
│
├── AGENT_STATE.md                     # Engineering lifecycle & phase tracking matrix
├── requirements.txt                   # Python dependencies
└── README.md                          # You are here!
```

---

## 🚀 Quickstart: Running the Engine Locally

### 1. Prerequisites
- Python 3.10 to 3.14
- Linux, macOS, or Windows WSL2 (Ubuntu recommended)

### 2. Set Up the Environment
Clone the repository and install dependencies in a virtual environment:

```bash
# Clone the repository
git clone https://github.com/Smart-Railways/railway-ai.git
cd railway-ai

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Automated Test Suite

```bash
pytest tests/ -v
```

---

## 💡 Using the Python API in 5 Lines

You can import and interact with the unified `RailwayMLEngine` directly from Python:

```python
import pandas as pd
from src.services.ml_engine import RailwayMLEngine

# 1. Initialize the engine
engine = RailwayMLEngine()

# 2. Check engine health and model availability
health = engine.health()
print(f"Engine status: {health['status']} (Models loaded: {health['models_loaded']})")

# 3. Create a sample worklist of pending maintenance tasks
tasks = pd.DataFrame([
    {
        "task_id": "TASK-001",
        "section_id": "NDL-MTJ-01",
        "department": "ENGINEERING",
        "condition_score": 38.0,
        "criticality": 4,
        "urgency": 5,
        "days_overdue": 14,
        "estimated_duration_hours": 3.0,
        "required_manpower": 12
    },
    {
        "task_id": "TASK-002",
        "section_id": "NDL-MTJ-01",
        "department": "S&T",
        "condition_score": 45.0,
        "criticality": 3,
        "urgency": 4,
        "days_overdue": 5,
        "estimated_duration_hours": 2.0,
        "required_manpower": 6
    }
])

# 4. Score tasks with calibrated ML failure risk and real corridor delay pressure
scored_tasks = engine.predict(tasks)
print(scored_tasks[["task_id", "maintenance_decision_score", "decision_reasons"]])

# 5. Generate a conflict-free block plan (weekly, monthly, rolling, or dynamic)
plan = engine.generate_block_plan(tasks, horizon_type="weekly")
print(f"Scheduled tasks across available windows!")

# 6. Generate Top-3 ranked feasible block window recommendations (Human-in-the-Loop)
recommendations = engine.recommend_windows(tasks)
print(f"Generated {recommendations['metrics']['total_recommendations']} ranked recommendations for human operator review!")
```

---

## 📊 Proof of Superiority: AI vs. Baseline

Does this mathematical optimization actually work better than traditional dispatching?

We built a strict benchmark validator (`BenchmarkValidator`) that compares the **Railway AI optimizer** directly against the standard industry baseline (**FIFO — First-In, First-Out Uncoordinated Scheduling**). 

The results across simulated corridors:
- **Critical Maintenance Throughput**: The AI optimizer schedules up to **25% more high-priority maintenance** within the exact same track availability limits.
- **Multi-Department Coordination**: The baseline achieves near-zero joint closures, while Railway AI groups **40%+ of adjacent tasks** into joint blocks, drastically cutting total track closures.
- **Delay Risk Reduction**: By steering track closures away from congested delay peaks, the AI minimizes knock-on delays to passenger trains.

---

## 🛡️ Project Principles & Governance

1. **Safety First**: Physical railway safety always overrides operational convenience. High-risk assets with dangerous condition scores receive maximum scheduling priority.
2. **Realistic Operations**: No synthetic or fictional station sequences. All corridor routes, station codes, and running times align with genuine Indian Railways operating realities.
3. **Preservation of Truth**: We maintain strict separation between synthetic research datasets and verified real-world operational evidence.

---

## 👥 Authors & Team

Built with ❤️ for safer, smarter, and more punctual railways by **Chirag Sharma** and the **Smart Railways** engineering team.
