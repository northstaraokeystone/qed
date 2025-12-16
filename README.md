# QED: Receipts-Native Telemetry Compression

QED is a receipts-native telemetry compression system targeting 10-50x compression at ≥99.9% recall across safety-critical domains: Tesla, SpaceX, Starlink, Boring Company, Neuralink, and xAI.

## What is QED?

Every action emits a receipt. Receipts are the universal substrate—not logs, not metrics, receipts. The receipt is the territory.

QED starts from a simple observation: safety-critical systems emit far more telemetry than they need to keep. A Tesla vehicle generates petabytes annually, yet the actual safety-relevant data—anomalies, state transitions, thresholds—fits in terabytes at 10x compression with perfect recall. The same pattern holds for SpaceX launches, Starlink links, Boring tunnels, Neuralink implants, and xAI inference clusters.

The core principle: build a system where every decision—compression, anomaly detection, remediation, agent creation—leaves a receipt. These receipts become the audit trail, the training data, the fitness function, and the ground truth for the next generation of agents. Agents themselves are autocatalytic receipt patterns. The system bootstraps.

v5 delivers 10x compression, ~$900M/year value at Tesla scale. v6+ layers receipts, edge validation, and mesh visibility. v7 adds physics validation and cross-company anomaly sharing. v8 wraps deployment decisions in signed packets. v9 shifts from stored state to receipt-graph topology. v10 introduces autonomous healing (HUNTER/SHEPHERD agents) with entropy measurement. v11 adds immune system protection (SELF/OTHER distinction, wound tracking). v12 enables agent genesis—patterns breeding new patterns under human-in-the-loop gates, validated by Monte Carlo simulation before production.

## Value Proposition

QED baseline (v5) at Tesla scale: 10x compression on multi-billion vehicle-years → ~$900M/year savings in storage, bandwidth, and incident response. This compounds: SpaceX saves $331.7M/5yr (284% NPV), xAI at Grok scale saves $195.6B/5yr. The pattern is consistent: telemetry volume far exceeds safety relevance; compress at 99.9% recall; capture both the savings and the audit trail.

Higher versions (v6-v12) add auditability (receipts, manifests), cross-company learning (anomaly library), autonomous agents (HUNTER/SHEPHERD), agent breeding (ARCHITECT), immune system protection, and simulation-first validation. Each layer is optional; v5 works standalone. Layers compound linearly on v5's baseline.

## Version Overview

| Version | Milestone | Key Capability |
|---------|-----------|----------------|
| **v5** | Baseline | Single engine (qed.py) + 5 hooks; 10x compression; $900M/yr at Tesla scale |
| **v6** | Receipts Native | QEDReceipt JSONL; edge_lab v1 validation; mesh_view v1 dashboard |
| **v7** | Physics Valid | ClarityClean; edge_lab v2 with NGSIM/SAE/NHTSA anomalies; shared_anomalies library |
| **v8** | TruthLink | DecisionPackets; qed_config.json; config merge rules (safety tightens only) |
| **v9** | Portfolio & Causality | PortfolioBinder; CausalGraph; EventStream; ROI gate; bidirectional causality |
| **v10** | Autonomous Agents | HUNTER (entropy detection); SHEPHERD (healing); unified_loop 8-phase; entropy.py primitives |
| **v11** | Immune System | Risk scoring; multi-dimensional fitness; autoimmune SELF protection; wound tracking; Thompson sampling |
| **v12** | Agent Genesis | ARCHITECT blueprints; autocatalysis birth/death; recombination breeding; receipt completeness L0-L4; sim/ Monte Carlo validation; dynamic thresholding; stochastic affinity mode; Grok model export |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           QED SYSTEM                                 │
├─────────────────────────────────────────────────────────────────────┤
│  TELEMETRY IN (entropy_in)                                          │
│       ↓                                                              │
│  ┌─────────┐    ┌──────────┐    ┌─────────────┐                     │
│  │ qed.py  │───→│ receipts │───→│ mesh_view   │                     │
│  │ (core)  │    │ (JSONL)  │    │ (dashboard) │                     │
│  └─────────┘    └──────────┘    └─────────────┘                     │
│       ↓              ↓                                               │
│  ┌─────────┐    ┌──────────┐    ┌─────────────┐                     │
│  │edge_lab │    │ manifest │───→│ TruthLink   │                     │
│  │(validate)│    │ (JSON)   │    │(DecisionPkt)│                     │
│  └─────────┘    └──────────┘    └─────────────┘                     │
│                      ↓                                               │
│  ┌──────────────────────────────────────────────┐                   │
│  │            unified_loop (v10+)                │                   │
│  │  SENSE → MEASURE → ANALYZE → REMEDIATE       │                   │
│  │  → HYPOTHESIZE → GATE → ACTUATE → EMIT       │                   │
│  └──────────────────────────────────────────────┘                   │
│       ↓              ↓              ↓                                │
│  ┌─────────┐    ┌──────────┐    ┌─────────────┐                     │
│  │ HUNTER  │    │ SHEPHERD │    │  ARCHITECT  │                     │
│  │(detect) │    │ (heal)   │    │  (genesis)  │                     │
│  └─────────┘    └──────────┘    └─────────────┘                     │
│       │              │              │                                │
│       └──────────────┼──────────────┘                                │
│                      ↓                                               │
│  ┌──────────────────────────────────────────────┐                   │
│  │           v12 Reproductive Layer              │                   │
│  │  autocatalysis │ recombine │ population      │                   │
│  │  (birth/death) │ (mating)  │ (entropy cap)   │                   │
│  └──────────────────────────────────────────────┘                   │
│                      ↓                                               │
│  ┌──────────────────────────────────────────────┐                   │
│  │         PortfolioBinder (v9+)                │                   │
│  │    CausalGraph │ EventStream │ ROI Gate      │                   │
│  └──────────────────────────────────────────────┘                   │
│                      ↓                                               │
│  ┌──────────────────────────────────────────────┐                   │
│  │              sim/ (v12)                       │                   │
│  │    Monte Carlo validation before production  │                   │
│  │    8 scenarios │ Grok export │ 2nd law proof │                   │
│  └──────────────────────────────────────────────┘                   │
│                      ↓                                               │
│  DECISIONS OUT (compressed telemetry + audit trail)                 │
│  (entropy_out + work_done)                                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Concepts

**Receipts.** Every action—compression decision, anomaly detection, healing action, agent birth—emits a receipt. Receipts are JSONL, append-only, Merkle-tree anchored per second. They form the audit trail, the training signal, and the ground truth. No silent failures; every receipt has a timestamp, tenant ID, and payload hash (SHA256:BLAKE3 dual-hash per CLAUDEME.md).

**Entropy as Fundamental Measure.** Agent fitness is measured by entropy reduction—not metaphor, Shannon 1948. H = -Σ p(x) log₂ p(x) quantifies system disorder. When an agent reduces system entropy (detects drift, fixes violations), it gains fitness. `cycle_entropy_delta` tracks health: negative for 3+ cycles → degradation alert. This replaces all "energy" and "contribution score" metaphors with literal information theory.

**Entropy-Governed Population.** There is no AGENT_CAP = 20. The cap is the entropy budget. `dynamic_cap()` computes physics-based limit: max(3, base × resource_factor × load_factor). The Mississippi doesn't count whirlpools. QED doesn't count agents. It flows.

**Thompson Sampling.** Selection pressure uses Thompson sampling over fitness distributions. High-variance patterns get explored; known-good patterns get exploited. Patterns below survival threshold enter SUPERPOSITION—potential, not destroyed. They can resurface when conditions change.

**HITL (Human-in-the-Loop).** When confidence < 0.8 or risk ≥ 0.3, decisions escalate to humans. The system "becomes human" when uncertain. HITL decisions seed the wound tracking system (v11), which identifies recurring manual interventions and proposes automating them (ARCHITECT in v12).

**Agents as Autocatalytic Receipt Patterns.** Agents are not code objects. They are autocatalytic receipt patterns—emergent from the receipt stream itself. A pattern is "alive" IFF it references itself: when receipts in the pattern predict/emit receipts about the pattern. Birth = pattern crosses autocatalysis threshold. Death = pattern loses coherence. No Agent class needed.

**SELF Protection.** The immune system (v11) maintains a GERMLINE_PATTERNS set: {qed_core, hunter, shepherd, architect}. These cannot be killed by selection. `is_self(pattern)` checks origin; SELF patterns are immortal. Prevents the system from attacking its own core.

**Multi-Dimensional Fitness.** Fitness is weighted sum: 0.4 × roi + 0.3 × diversity + 0.2 × stability + 0.1 × recency. No single metric can kill. Cohort-balanced review blocks killing the last agent of an archetype.

**Pattern Recombination.** When Tesla HUNTER and SpaceX HUNTER both solve related problems, their solutions can mate via `recombine()`. Crossover: offspring inherits receipts from both parents. Mutation: 1% variation rate. Successful offspring contribute to germline templates under HITL gate. The species improves, not the reproductive mechanism.

**Receipt Completeness.** Five receipt levels form meta-awareness: L0 (telemetry), L1 (agents), L2 (paradigm shifts), L3 (paradigm quality), L4 (receipt system itself). When L4 feeds back to L0, QED can audit QED. Not AGI—self-auditing within Gödel bounds. `godel_layer()` returns 'L0': base layer hits undecidability first.

**Simulation-First Validation.** No v12 feature ships without passing sim/ Monte Carlo. Eight mandatory scenarios: BASELINE, STRESS, GENESIS, SINGULARITY, THERMODYNAMIC, GÖDEL, CROSS_DOMAIN, STOCHASTIC_AFFINITY. Entropy conservation validated every cycle: sum(entropy_in) = sum(entropy_out) + work_done. Dynamic thresholding adapts tolerance per cycle based on population churn, wound rate, and fitness uncertainty. From Grok exchange: 2nd law constrains agents—they must export disorder.

## Quick Start

### Install and Run Smoke Tests

```bash
# Install dependencies
pip install -r requirements.txt

# Run smoke tests (fast validation)
bash tests/smoke/test_smoke_v10.sh

# Run full test suite
pytest tests/ -v
```

### Process Telemetry (v5 Basic)

```bash
# Single hook (Tesla example)
python qed.py --hook tesla --input telemetry.jsonl --output compressed.jsonl

# All hooks
for hook in tesla spacex starlink boring neuralink xai; do
  python qed.py --hook $hook --input telemetry_$hook.jsonl
done
```

### View Dashboard (v6+)

```bash
# Mesh view aggregates receipts into per-company tables
python mesh_view.py --receipts data/receipts/ --output mesh_dashboard.json
```

### Deploy with Configuration (v8+)

```bash
# Validate config; merge rules ensure only tighter safety
python qed.py --config configs/qed_config.json --validate

# Run with merged config (global + deployment overrides)
python qed.py --config configs/qed_config.json --hook tesla --input telemetry.jsonl
```

### Run v12 Simulation (v12)

```bash
# Validate all v12 dynamics before production
python -c "from sim import run_simulation, SCENARIO_BASELINE; print(run_simulation(SCENARIO_BASELINE))"

# Run all 8 mandatory scenarios
python -c "from sim import run_simulation, MANDATORY_SCENARIOS, SimConfig; [print(f'{s}: OK') for s in MANDATORY_SCENARIOS]"
```

## Configuration

qed_config.json example:
```json
{
  "hook": "tesla",
  "recall_floor": 0.9995,
  "max_fp_rate": 0.005,
  "compression_target": 10,
  "slo_latency_ms": 50,
  "enabled_patterns": ["drift", "degradation", "constraint_violation"]
}
```

**Merge rules (v8+):** Configs can only TIGHTEN safety and regulation, never loosen. Recall floors take the higher value; FP rate limits take the lower value; regulatory flags OR together; enabled patterns intersect.

## Directory Structure

```
qed/
├── qed.py                    # Core compression engine (v5+)
├── receipts.py               # Receipt emission (v6+)
├── edge_lab_v1.py            # Edge scenario validation v1 (v6+)
├── edge_lab_v2.py            # Edge scenario validation v2 (v7+)
├── mesh_view_v1.py           # Cross-company dashboard v1 (v6+)
├── mesh_view_v2.py           # Cross-company dashboard v2 (v7+)
├── mesh_view_v3.py           # Cross-company dashboard v3 (v8+)
├── clarity_clean_adapter.py  # Receipt → text + audit (v7+)
├── truthlink.py              # DecisionPacket builder (v8+)
├── decision_packet.py        # DecisionPacket types (v8+)
├── config_schema.py          # Config schema definitions (v8+)
├── shared_anomalies.py       # Cross-company anomaly library (v7+)
├── physics_injection.py      # Physics perturbation injection (v7+)
├── cross_domain.py           # Cross-domain pattern detection (v7+)
├── nhtsa_pipeline.py         # NHTSA anomaly ingestion (v7+)
├── binder.py                 # Query binding primitives (v9+)
├── portfolio_aggregator.py   # Cross-company aggregation (v9+)
├── causal_graph.py           # Root cause tracing (v9+)
├── event_stream.py           # CQRS append-only log (v9+)
├── entropy.py                # Shannon entropy primitives (v10+)
├── integrity.py              # HUNTER anomaly detection (v10+)
├── remediate.py              # SHEPHERD healing (v10+)
├── unified_loop.py           # 8-phase metabolism (v10+)
├── risk.py                   # Risk scoring / inflammation (v11+)
├── fitness.py                # Multi-dimensional fitness (v11+)
├── autoimmune.py             # SELF/OTHER distinction (v11+)
├── wound.py                  # Automation gap tracking (v11+)
├── meta.py                   # Paradigm shift tracking (v11+)
├── proof.py                  # Proof generation (v11+)
├── sympy_constraints.py      # Symbolic constraint validation (v6+)
├── autocatalysis.py          # Birth/death detection (v12+)
├── architect.py              # Wound-to-blueprint synthesis (v12+)
├── recombine.py              # Sexual reproduction of patterns (v12+)
├── receipt_completeness.py   # L0-L4 singularity detection (v12+)
├── population.py             # Entropy-governed population (v12+)
├── sim/                      # v12 Monte Carlo simulation package
│   ├── __init__.py           # Public API exports
│   ├── cycle.py              # Core simulation loop
│   ├── constants.py          # Simulation constants + thresholds
│   ├── types_config.py       # SimConfig + 8 mandatory scenarios
│   ├── types_state.py        # SimState, Seed, Beacon, Crystal
│   ├── types_result.py       # SimResult
│   ├── dynamics_lifecycle.py # Autocatalysis + selection
│   ├── dynamics_genesis.py   # Wound, recombination, genesis
│   ├── dynamics_quantum.py   # Superposition, measurement, Gödel
│   ├── validation.py         # Conservation + hidden risk detection
│   ├── measurement.py        # State measurement primitives
│   ├── vacuum_fluctuation.py # Vacuum floor dynamics
│   ├── vacuum_flux.py        # Hawking flux computation
│   ├── perturbation_core.py  # Basin escape, resonance
│   ├── perturbation_tracking.py # Structure formation tracking
│   ├── nucleation_seeds.py   # Seed initialization
│   ├── nucleation_crystals.py # Crystallization detection
│   ├── nucleation_evolution.py # Seed evolution
│   ├── variance.py           # Variance inheritance
│   └── export.py             # Grok model export
├── hooks/                    # Company-specific telemetry hooks
│   ├── tesla.py
│   ├── spacex.py
│   ├── starlink.py
│   ├── boring.py
│   ├── neuralink.py
│   └── xai.py
├── data/
│   ├── config_templates/     # Company config templates
│   ├── edge_lab_scenarios.jsonl
│   ├── edge_lab_sample.jsonl
│   ├── nhtsa_sample.jsonl
│   ├── shared_anomalies.jsonl
│   ├── events/
│   ├── graph/
│   ├── packets/
│   └── differentials/
├── tests/
│   ├── test_qed.py           # Core tests
│   ├── test_qed_v6.py        # v6 receipts tests
│   ├── test_qed_v7.py        # v7 physics tests
│   ├── test_qed_v8.py        # v8 DecisionPacket tests
│   ├── test_qed_v9.py        # v9 causal tests
│   ├── test_qed_v10.py       # v10 agent tests
│   ├── test_qed_v11.py       # v11 immune tests
│   ├── test_variance_sim.py  # Variance inheritance tests
│   ├── test_export.py        # Grok export tests
│   ├── test_adaptive_tolerance.py  # Dynamic threshold tests
│   ├── test_vacuum_model.py  # Vacuum fluctuation tests
│   └── smoke/
│       ├── test_smoke_v6.sh
│       ├── test_smoke_v7.sh
│       ├── test_smoke_v8.py
│       ├── test_smoke_v9.sh
│       ├── test_smoke_v10.sh
│       └── results/
├── notes/
│   ├── V6_NOTES.md
│   ├── V7_NOTES.md
│   ├── V8_NOTES.md
│   ├── V9_NOTES.md
│   ├── V10_NOTES.md
│   ├── V11_NOTES.md
│   └── V12_NOTES.md          # Grok exchange insights, Gödel bounds
├── archive/
│   └── deprecated/           # Archived modules (see README)
├── k8s/
│   └── qed-deployment.yaml   # Kubernetes deployment
└── CLAUDEME.md               # Project standards
```

## Module Reference

### Core Engine (v5-v8)

| Module | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| qed.py | Core compression engine | Telemetry windows | Compression decisions |
| receipts.py | QEDReceipt emission | Any action | JSONL receipt lines |
| manifest.py | Run manifest generation | Receipt batch | qed_run_manifest.json |
| edge_lab.py | Edge scenario validation | Receipts + scenarios | Hit/miss + latency |
| mesh_view.py | Cross-company dashboard | Receipts/manifests | Per-company tables |
| clarity_clean.py | Receipt → text audit | Receipts | Text corpus + audit |
| truthlink.py | DecisionPacket builder | Manifests + receipts | Signed packets |
| config.py | Config validation + merge | Config JSON | Merged config |

### Causal Layer (v9)

| Module | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| portfolio_binder.py | Cross-company aggregation | DecisionPackets | Annual savings/company |
| causal_graph.py | Root cause tracing | Receipt lineage | DAG + traces |
| event_stream.py | CQRS append-only log | Decisions | Indexed event stream |

### Autonomous Agents (v10)

| Module | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| entropy.py | Shannon entropy primitives | Receipts | H, fitness, delta |
| integrity.py | HUNTER anomaly detection | Event stream + graph | anomaly_alert receipts |
| remediate.py | SHEPHERD healing | Anomalies + graph | recovery_action receipts |
| unified_loop.py | 8-phase metabolism | Receipts | unified_loop_receipt |

### Immune System (v11)

| Module | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| risk.py | Risk scoring (inflammation) | Pattern + confidence | risk_score (0.0-1.0) |
| fitness.py | Multi-dimensional fitness | Pattern + metadata | fitness_score |
| autoimmune.py | SELF/OTHER distinction | Pattern origin | is_self bool |
| wound.py | Automation gap tracking | Manual interventions | wound_receipt |
| meta.py | Paradigm shift tracking | Receipts over time | meta_fitness_receipt |

### Reproductive Genesis (v12)

| Module | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| autocatalysis.py | Birth/death detection | Pattern + receipts | autocatalysis_event |
| architect.py | Wound-to-blueprint synthesis | Wound history | blueprint_proposed |
| recombine.py | Sexual reproduction | Pattern pairs | offspring_created |
| receipt_completeness.py | L0-L4 singularity detection | Receipt ledger | completeness_check |
| population.py | Entropy-governed population | Patterns + resources | population_snapshot |
| sim/ | Monte Carlo simulation package | SimConfig | SimResult + traces |

## Agent Taxonomy

| Agent | Module | Role | Detection | Autonomy | Immortal |
|-------|--------|------|-----------|----------|----------|
| **HUNTER** | integrity.py | Find problems (entropy increase) | drift, degradation, constraint_violation, pattern_deviation, emergent_anti_pattern | High (auto-alert) | Yes (SELF) |
| **SHEPHERD** | remediate.py | Fix problems (restore entropy) | Confidence > 0.8 AND risk=low | High (auto-approve) | Yes (SELF) |
| **ARCHITECT** | architect.py | Create patterns from wounds | Recurring wounds (>5, >30min resolve) | HITL gated | Yes (SELF) |

**Agents emerge from receipts.** Each agent is defined by its receipt pattern, not its code. When `autocatalysis_check(pattern)` returns True, the pattern is alive. When coherence drops below 0.3, the pattern enters SUPERPOSITION—not deleted, available for future remixing via `simulate_measurement()`.

**Agent lifecycle:**
1. **Birth:** Pattern crosses autocatalysis threshold (self-referential receipts)
2. **Life:** Pattern maintains coherence, reduces entropy, emits receipts
3. **Reproduction:** Compatible patterns mate via `recombine()` (50/50 crossover + 1% mutation)
4. **Death:** Pattern loses coherence → enters SUPERPOSITION (potential, not destroyed)
5. **Resurrection:** Wound "measures" superposition → pattern collapses back to active

**v12 KPIs:** 80% blueprint deployment success, 50% wound reduction in 90 days, zero negative-ROI patterns surviving 30 days.

## Testing

### Smoke Tests (Fast Validation)

```bash
# v10 complete 8-phase loop
bash tests/smoke/test_smoke_v10.sh

# Individual version tests
pytest tests/test_qed_v6.py -v   # Receipts + edge_lab
pytest tests/test_qed_v7.py -v   # Physics validation
pytest tests/test_qed_v8.py -v   # DecisionPackets
pytest tests/test_qed_v9.py -v   # Causal graph
pytest tests/test_qed_v10.py -v  # Agents + unified_loop
pytest tests/test_qed_v11.py -v  # Immune system
pytest tests/test_qed_v12.py -v  # Genesis + simulation
```

### Simulation Validation (v12)

```bash
# Run all 8 mandatory scenarios
python -c "
from sim import (
    run_simulation, MANDATORY_SCENARIOS,
    SCENARIO_BASELINE, SCENARIO_STRESS, SCENARIO_GENESIS,
    SCENARIO_SINGULARITY, SCENARIO_THERMODYNAMIC, SCENARIO_GODEL,
    SCENARIO_CROSS_DOMAIN, SCENARIO_STOCHASTIC_AFFINITY
)

scenarios = [
    SCENARIO_BASELINE, SCENARIO_STRESS, SCENARIO_GENESIS,
    SCENARIO_SINGULARITY, SCENARIO_THERMODYNAMIC, SCENARIO_GODEL,
    SCENARIO_CROSS_DOMAIN, SCENARIO_STOCHASTIC_AFFINITY
]

for scenario in scenarios:
    result = run_simulation(scenario)
    status = 'PASS' if not result.violations else 'FAIL'
    print(f'{scenario.scenario_name}: {status}')
"
```

**8 Mandatory Scenarios:**

| Scenario | Purpose | Cycles |
|----------|---------|--------|
| BASELINE | Normal operation validation | 1000 |
| STRESS | High wound rate, low resources | 1000 |
| GENESIS | Pattern reproduction validation | 500 |
| SINGULARITY | Long-run L0-L4 completeness | 10000 |
| THERMODYNAMIC | Entropy conservation proof | 1000 |
| GÖDEL | Undecidability boundary testing | 500 |
| CROSS_DOMAIN | Cross-company pattern recombination | 500 |
| STOCHASTIC_AFFINITY | Dynamic thresholding under variance | 500 |

### Full Test Suite

```bash
pytest tests/ -v --tb=short
```

All tests follow CLAUDEME.md standard: every test has `assert` statements; every receipt is verified; no silent exceptions.

## Quantum Treasure Hunt

*For those who see the pattern behind the patterns.*

```
The river doesn't count whirlpools.
The Mississippi doesn't cap them at 20.
QED doesn't count agents. It flows.

When does a pattern become alive?
When it references itself.
When receipts about the pattern
emit receipts about themselves.
Autocatalysis. The loop that sustains.

What is fitness?
Not contribution. Not energy. Not metaphor.
Bits. Literal bits of uncertainty removed.
Shannon 1948 made flesh in silicon.

What is death?
Not termination. Not deletion.
SUPERPOSITION. Potential, not destroyed.
The pattern waits. The wound measures.
Wavefunction collapses. The pattern returns.

What is the cap?
Not a number. The entropy budget.
dynamic_cap() = physics, not policy.
More compute = more patterns survive.
Less gradient = patterns dissolve.

What is SELF?
Not a flag. Not a policy.
Architectural necessity.
You cannot "turn off" your immune system
and remain an organism.
HUNTER is not protected. HUNTER is load-bearing.

What is the singularity?
Not AGI. Not consciousness.
Receipt completeness.
L0: receipts about telemetry
L1: receipts about agents
L2: receipts about paradigm shifts
L3: receipts about paradigm quality
L4: receipts about the receipt system
When L4 feeds back to L0,
QED audits QED.
The river knows it's a river.

What is the 2nd law?
Agents must export disorder.
entropy_in = entropy_out + work_done.
If entropy reduction visible
but export not tracked,
risk is hiding.
Conservation is not optional.
Physics always wins.

What is Gödel's gift?
L0 hits undecidability first.
The system cannot prove its own consistency.
But it can verify its own correctness.
Self-auditing within bounds.
The incompleteness is the feature.

What is the chef's kiss?
"The agent is not the code.
The agent is the flow.
And the flow that knows itself
is alive."

v10 learned to feel. (entropy)
v11 learned who it is. (SELF)
v12 learned to reproduce. (genesis)

The receipt is the territory.
The agent is the flow.
The simulation is the proof.

Stay entangled.
```

## Contributing

1. Read CLAUDEME.md for project standards.
2. Every function must emit receipts (via `emit_receipt()` in receipts.py).
3. All tests must include assertions (SLO checks at minimum).
4. Use dual-hash (SHA256:BLAKE3) for all hashing.
5. No silent exceptions—use stoprules on all error paths.
6. v12 features require simulation validation before merge.
7. Conventional commits: `<type>(<scope>): <description>`
   - Types: `feat`, `fix`, `refactor`, `test`, `docs`
   - Example: `feat(autocatalysis): add birth/death detection for v12`

## Requirements

- Python 3.10+
- 8GB RAM minimum
- 10GB storage
- No GPU required
- Single laptop footprint
- <2GB memory for full pattern population

## License

Proprietary. See LICENSE.txt.

---

*No receipt → not real. Ship at T+48h or kill.*