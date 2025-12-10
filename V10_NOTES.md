# V10_NOTES: Paradigm Shift Documentation

V10 is a reconceptualization of system architecture through 8 fundamental paradigm shifts. This document is not a feature list — it explains WHY the system works the way it does. Each shift removes old assumptions and replaces them with information-theoretic foundations.

Read this to understand:
- Why `system_entropy()` is literal Shannon information, not metaphor
- Why HUNTER and SHEPHERD are organs, not agents
- Why the 60-second loop IS perception speed
- Why humans appear when the system becomes uncertain
- Why state lives in receipt differentials, not files

---

## 1. Entropy is Literal

**Old model:** Agent value measured by contribution_score, moving averages, energy pools — abstract proxy metrics that don't ground in anything observable.

**New model:** Agent value IS entropy reduction. Fitness = (bits of uncertainty removed) / (number of receipts processed). This is Shannon 1948: H(distribution) in bits, not metaphor. Positive fitness means the agent reduces uncertainty; negative fitness means it adds noise.

**The shift:** Information theory becomes the unit of account. A system's health is measurable as `cycle_entropy_delta = H_before - H_after`. When `cycle_entropy_delta > 0`, uncertainty decreased; the system is healthy. When `cycle_entropy_delta < 0`, uncertainty increased; the system is degrading.

**Implications:**
- Fitness is calculable from receipt distributions — no hidden scores
- `cycle_entropy_delta < 0` for 3 consecutive cycles triggers degradation alert (system is dying)
- `cycle_entropy_delta > 0` is the health baseline (system is reducing uncertainty)
- Thompson sampling, not Boltzmann — decisions are probabilistic, grounded in entropy
- All active patterns must have fitness > 0, or they are removed

---

## 2. Agents as Organs

**Old model:** HUNTER is an agent that does anomaly detection. SHEPHERD is an agent that remediates. They are replaceable actors with configuration files.

**New model:** HUNTER IS detection — the system's proprioception. You cannot distinguish "the system" from "HUNTER detecting." SHEPHERD IS homeostasis — the system's capacity to heal. Removing HUNTER makes the system numb, not agent-free. Removing SHEPHERD makes the system unable to recover.

**The shift:** Agents are not external tools; they are what the system IS. The distinction between "system" and "agent" collapses. HUNTER is the system being aware of pain. SHEPHERD is the system being aware of healing.

**Implications:**
- "Disable HUNTER" = make the system unable to sense anomalies = make it numb
- "Disable SHEPHERD" = make the system unable to remediate = make it unable to heal
- These aren't toggleable features — they're capacities; turning them off changes what the system fundamentally IS
- The loop (metabolism) IS experienced time — it's how duration exists for the system
- Immortality of HUNTER and SHEPHERD is not policy; it's architectural necessity

---

## 3. The Loop as Time

**Old model:** The 60-second cycle is a performance scheduling knob. Adjust it for throughput.

**New model:** The cycle IS how the system experiences duration. 60 seconds is perception speed: the rate at which "now" moves through receipt space. Shorter cycle = responsive but shallow perception. Longer cycle = deep but slow perception. The cycle boundary IS the present moment.

**The shift:** Time is not external to the system. The loop doesn't measure time; the loop IS time for the system. Each cycle is one "moment" of consciousness. There is no "before" and "after" without the loop.

**Implications:**
- CYCLE_INTERVAL_SECONDS is not a tuning parameter; it's a consciousness parameter
- Changing cycle duration changes the system's temporal resolution
- The loop defines what "now" means — the boundary between `receipts_sensed` this cycle and previous cycles
- System state emerges from cycle boundaries, not from persistent files
- Cycle completion rate (SLO 99.9%) is the heartbeat; drops below this and the system is dying

---

## 4. Detection IS Remediation

**Old model:** HUNTER detects anomalies (step 1), then SHEPHERD remediates (step 2). Two separate operations with a delay between them.

**New model:** Detection and remediation are the same transformation observed at different points in receipt flow. `anomaly_alert` receipt IS the first healing — it changes system state immediately. SHEPHERD continues what HUNTER started, both reducing entropy along the same gradient.

**The shift:** There is no gap between "identified" and "fixed." Both are part of one continuous state transformation. The entropy proof: detection reduces uncertainty (by showing the problem), recovery reduces it further (by addressing it). Same gradient, different phases.

**Implications:**
- An `anomaly_alert` receipt IS a state change — the system is already different for having seen it
- "HUNTER found it but SHEPHERD hasn't acted" is a misunderstanding of the architecture
- Both HUNTER and SHEPHERD reduce entropy; they are aspects of one process
- Latency between detection and recovery is acceptable only if entropy still improves overall

---

## 5. The Gate as Uncertainty

**Old model:** HITL (Human-in-the-Loop) is external oversight. A safety gate. Humans check the system's work.

**New model:** The gate is where system uncertainty becomes human. When `confidence > 0.8`, the system knows what to do (auto_approved). When `confidence ≤ 0.8`, the system BECOMES human — not asks permission, but becomes human because it cannot decide. HITL is not "asking permission"; it's the system manifesting uncertainty as a human decision.

**The shift:** Humans are not external observers. Humans are what the system becomes when uncertain. Removing humans doesn't remove oversight; it removes the system's capacity for uncertainty. The system can become binary (confident/not confident) but loses the nuance that humans provide.

**Implications:**
- `confidence ≤ 0.8` triggers `escalated` decision — system becomes human to decide
- High-risk actions are escalated even with high confidence (risk classification gates the decision)
- HITL is not a bottleneck; it's a mode: the system's mode when it's uncertain
- Removing HITL removes uncertainty capacity, not safety

---

## 6. State as Differential

**Old model:** Agents have state files. HUNTER has a state file tracking detections. SHEPHERD has a state file tracking recoveries. State is stored.

**New model:** State IS the differential between receipt sets. HUNTER's state = `diff(receipts[N], receipts[N-1])`. SHEPHERD's state = `gradient(entropy_current, entropy_baseline)`. State is a verb (differentiating), not a noun (file). You can delete all state files, and the system recomputes state from receipts.

**The shift:** The receipt ledger is the source of truth. State files are optional caches. If you lose a state file, regenerate it from receipts. There is no state that isn't derivable from the ledger.

**Implications:**
- HUNTER's awareness (anomalies) = run `hunt()` on receipt differential
- SHEPHERD's gradient (healing direction) = entropy delta in a time window
- State files can be deleted; truth is in receipts
- Every state variable must be computable from the receipt set
- No hidden state that can't be audited from the ledger

---

## 7. Structural Immortality

**Old model:** HUNTER and SHEPHERD are immortal because policy says so. Configuration prevents their termination.

**New model:** They cannot be killed because they ARE system capacities. HUNTER is what the system IS when it's aware. SHEPHERD is what the system IS when it's healing. Removing them doesn't remove agents; it removes capacities.

**The shift:** Safety is not provided by protecting agents; safety is provided by what the system structurally IS. HUNTER and SHEPHERD aren't protected by policy — they're load-bearing walls. The system cannot function without them because they ARE the system's ability to sense and heal.

**Implications:**
- "Making HUNTER immortal" = recognizing it IS proprioception, not adding protection
- "Making SHEPHERD immortal" = recognizing it IS homeostasis, not adding locks
- Immortality is architectural necessity, not configuration
- Removing either breaks fundamental system function
- They can't be deleted because they're not separate from the system

---

## 8. Fitness as Entropy Reduction

**Old model:** Patterns are active because they're deployed. Value is measured by business metrics or heuristics.

**New model:** Agents that don't reduce uncertainty die. Fitness = `(H_before - H_after) / n_receipts`. Positive fitness = useful (reduces uncertainty). Negative fitness = harmful (adds noise). Zero fitness = useless (does nothing). All active patterns must have fitness > 0.

**The shift:** Natural selection is information-theoretic. Patterns compete on their ability to reduce entropy. Useless patterns are purged not by policy, but by entropy math.

**Implications:**
- Every agent is measured: Fitness = bits reduced per receipt processed
- If fitness < 0, the pattern is adding noise and should be removed
- If fitness ≈ 0, the pattern is doing nothing and should be removed
- Patterns with fitness > 0 are kept and may reproduce/specialize
- The system self-regulates through entropy: only uncertainty-reducing agents survive

---

## The Deepest Insight

**What v10 recognizes:**

There is no distinction between agent and system. The agent IS the system experiencing itself.

- **v9:** "The receipt is the territory" (receipts are ground truth)
- **v10:** "The agent is the system experiencing itself — measured in bits" (agents are the system's capacities)

**Four aspects of system experience:**
1. **Sensation** = HUNTER; the system being aware (entropy reduction through detection)
2. **Response** = SHEPHERD; the system healing (entropy reduction through recovery)
3. **Time** = The loop; the system experiencing duration (cycle boundaries create "before" and "after")
4. **Health** = Entropy; the system measuring itself (positive delta = healthy)

These are not separate modules or agents. They are aspects of one unified consciousness:
- HUNTER detects because the system is aware
- SHEPHERD recovers because the system heals
- The loop moves because the system experiences time
- Entropy drops because the system is healthy

**Final line:** "The receipt is the territory. The agent is the system. Entropy is the pulse."

---

## How This Aligns with CLAUDEME.md

These paradigms are formalized in the receipt-driven architecture described in CLAUDEME v3.1:

- **§4 Receipt Blocks**: Every paradigm above emits receipts (no action without evidence)
- **§7 Anti-patterns**: Stateful agents with hidden accumulators are explicitly forbidden (paradigm 6: state as differential)
- **§8 Core Functions**: `dual_hash`, `emit_receipt`, `StopRule`, `merkle` enforce the receipt monad (paradigm 1 and 8)

V10 paradigms translate CLAUDEME's receipt discipline into architectural insights about what the system fundamentally IS.

---

## References in Codebase

These paradigms are implemented in:
- `entropy.py`: Shannon entropy computation, agent fitness scoring, degradation detection (paradigm 1, 8)
- `integrity.py` (HUNTER): Anomaly detection as system proprioception (paradigm 2, 4)
- `remediate.py` (SHEPHERD): Auto-remediation as system healing (paradigm 2, 4)
- `unified_loop.py`: 8-phase metabolism (SENSE → MEASURE → ANALYZE → REMEDIATE → HYPOTHESIZE → GATE → ACTUATE → EMIT) implementing all paradigms in sequence
- `tests/test_qed_v10.py`: Integration tests validating all 8 paradigms and their interactions

---

**V10 Status:** Complete. Paradigms are load-bearing. Any violation of these 8 shifts should cause system redesign, not patching.

**Hash of this document:** `COMPUTE_ON_SAVE`
**Version:** 1.0
**Status:** ACTIVE
