# V9_NOTES: Paradigm Shifts in QED v9

V9 is a paradigm shift measured by deletions, not additions. Each deletion removes one degree of freedom to gain clarity. Six systems were rearchitected; the manifesto below documents their evolution and death conditions.

---

## 1. The Receipt Monad

**Core Question**: Why R → R signature on all modules?

### What was deleted
- Stateful services with hidden accumulators (counts, rolling windows, state machines)
- Implicit side effects (logging directly to disk, modifying shared cache without receipt)
- Module-to-module contracts that relied on shared memory or hidden preconditions

### What remains
- Pure function: `transform(List[Receipt]) → List[Receipt]`
- Every input receipt and output receipt is explicit, hashable, and addressable
- Replay is free: feed the same receipts in, get identical receipts out
- Testing is direct: no mocks, no fixtures, no setup/teardown; just call the function

### One-line test
Input receipts with known ingestion_timestamp and payload_hash; verify output receipts have deterministic signatures that match when replayed.

### Why this matters (and its limits)
The Receipt Monad forces every side effect—including metadata collection—to appear as a receipt in the output. This eliminates the "hidden state" problem that makes distributed systems hard to audit. However, the monad does not solve latency or resource cost; it only makes them visible. A slow pure function is still slow.

---

## 2. Value as Topology

**Core Question**: Why delete dollar_value_annual and compute centrality instead?

### What was deleted
- Stored `dollar_value_annual` field in pattern records
- Hard-coded financial estimates baked into deployment configs
- Assumption that a pattern's value is static across deployments or time windows

### What remains
- Compute value at query time from receipt graph structure: PageRank centrality of pattern nodes
- A pattern's value emerges from how many receipts depend on it, how critical those dependents are, and how deep the dependency chain
- Financial impact is derived, not dictated

### One-line test
Query the same pattern at two different times with different receipts; centrality scores differ deterministically based on the graph structure at query time.

### Why this matters (and its limits)
Value is not intrinsic; it is relational. A critical safety pattern in one fleet can be noise in another. Computing value from topology means deployments don't need to pre-declare their ROI; the system observes it. This breaks the assumption that financial metrics are stable, which is both powerful and dangerous: a small receipt deletion can cascade and change all pattern valuations. Auditors must now learn graph theory.

---

## 3. Mode Elimination

**Core Question**: Why delete PatternMode enum?

### What was deleted
- Enum with states: LIVE, SHADOW, DEPRECATED
- Metadata field indicating deployment state
- Assumption that a pattern has a single, globally visible lifecycle state

### What remains
- Mode becomes a predicate: `query(pattern, actionable=True)` returns only patterns that pass a set of filters
- Same pattern can be LIVE for one observer (meeting their coverage thresholds), SHADOW for another (under trial), and invisible to a third (excluded by policy)
- Mode is context-dependent; queried, not stored

### One-line test
Same pattern queried by two observers with different filter predicates; one receives it as LIVE, the other as SHADOW, with identical underlying data.

### Why this matters (and its limits)
This decouples global metadata from local visibility. It is powerful for multi-tenant and multi-deployment scenarios where a pattern's role varies. It also means there is no single source of truth about pattern state; each observer constructs their own view. This is simpler but requires strict predicate discipline: a buggy filter can silently hide critical patterns.

---

## 4. Bidirectional Causality

**Core Question**: Why replace DAG with flow network?

### What was deleted
- Acyclic graph assumption: pattern A → incident B → loss C
- Unidirectional trace_forward() only
- Assumption that causality has a fixed time direction

### What remains
- Causal graph is now a flow network with bidirectional edges
- `trace_forward(pattern)` asks "what happens if this pattern fires?"
- `trace_backward(incident)` asks "what patterns could have prevented this?"
- `counterfactual(threshold_delta, t=-7days)` replays the graph with changed parameters to ask "what if we had tuned this in the past?"

### One-line test
For a historical incident, backward-trace to root patterns, perturb one pattern's threshold, replay forward with new threshold, verify the incident would not have occurred.

### Why this matters (and its limits)
DAGs are causal folklore; in reality, feedback loops abound. Pattern A detects anomaly B, which triggers mitigation C, which prevents pattern A from firing again. Bidirectionality makes this visible. The catch: counterfactual replay assumes the model is complete and correct. If the model is missing a feedback loop or misses a confound, the replay is wrong. Every "what if" must be checked against reality.

---

## 5. Self-Referential Compression

**Core Question**: Why is self_compression_ratio a health metric?

### What was deleted
- Ad-hoc health checks (CPU, disk, latency percentiles)
- Health dashboards with 50+ independent metrics
- Assumption that system health is an external property to be measured

### What remains
- One metric: `self_compression_ratio = (receipts_emitted / receipts_needed_to_reconstruct) → [0, 1]`
- Measure how well the system can reconstruct its own behavior from its receipt ledger
- High compression (e.g., 0.92) = simple self-model = healthy; sudden drops (e.g., 0.62) = anomaly, restart

### One-line test
At query time, sample 100 random historical receipts and verify they can be reconstructed from the graph; compression_ratio = hit_count / 100.

### Why this matters (and its limits)
This is Occam's Razor automated. A healthy system should be able to explain itself parsimoniously. If you need many receipts to reconstruct what happened, something is off. The weakness: compression can be high even when behavior is wrong, if the wrongness is systematic and consistent. A biased system that always makes the same error will have high compression.

---

## 6. Entanglement Over Aggregation

**Core Question**: Why is sum aggregation wrong?

### What was deleted
- Portfolio arithmetic: pattern_value_total = sum(pattern_value[i] for all deployments)
- Assumption that effects are linear and independent across deployments
- Stored aggregation fields (sum_centrality, avg_false_positive_rate)

### What remains
- Entanglement coefficient: observing Tesla's pattern P instantaneously affects SpaceX's inference about P
- Cross-deployment correlation is computed, not pre-aggregated
- Portfolio view = weighted entanglement graph, not sum of values

### One-line test
Freeze the graph; add one Tesla receipt for pattern P; recompute SpaceX's belief about P; verify the belief changed even though SpaceX had no new receipt.

### Why this matters (and its limits)
Patterns are not independent. One company's data breach affects the risk profile of a shared infrastructure component. Summing assumes independence, which is false in entangled systems. This is quantum-inspired and breaks linear intuition. The catch: entanglement is bidirectional and can create feedback loops where observing one deployment changes another's view indefinitely. The system must have a fixed-point solver to avoid infinite loops.

---

## 7. Unfinished Manifesto

V9 is not finished. It is alive and should remain so as long as the following constraints hold. When any constraint breaks, v9 should be deleted, not patched.

### open_questions

1. **Graph divergence under concurrent updates**: If two deployments add receipts simultaneously for the same pattern, can graph centrality differ depending on merge order? Proof that centrality is commutative under receipt merge is missing.

2. **Entanglement fixed-point existence**: Under which conditions does `entanglement_coefficient()` converge for arbitrary deployment graphs? We have no theorem. Empirically it converges in <10ms for our test fleets, but a pathological graph topology might diverge.

3. **Counterfactual fidelity**: When we replay with a changed threshold from 7 days ago, how do we know we modeled all the feedback loops? If the model is missing a closed loop (e.g., auto-escalation logic), the replay is fiction. We have no principled test for model completeness.

4. **Cross-domain pattern transfer**: Can a pattern learned in Tesla (vehicle telemetry) meaningfully apply to SpaceX (launch dynamics) or Boring (tunnel boring)? We assume yes; no empirical validation exists.

5. **Self-compression as divergence detector**: High compression might mask systematic bias. A fraudulent system with consistent behavior will compress well. Are we trading robustness for parsimony?

### entropy_hooks

These are deliberate, receipted slots for future experiments. They are OFF by default and must be explicitly enabled with an entropy_receipt.

1. **DragonChaos::entanglement_randomization** – Inject random noise into entanglement coefficients to test if the system degrades gracefully. If performance collapses, entanglement is fragile. Intended use: once per quarter in shadow, with receipt-gated rollback.

2. **DragonChaos::counterfactual_divergence_bounds** – Run counterfactual with two different replay engines and compute Wasserstein distance between resulting incident predictions. If divergence is high, model is under-specified. Shadow-only; must not affect production queries.

3. **DragonChaos::pattern_value_rewiring** – Temporarily rewire the centrality graph to exclude one deployment and re-query all patterns. Measure how much pattern valuations shift. If shifts are large, system is fragile to deployment loss. Periodic audit (monthly) with receipts.

4. **DragonChaos::compression_malefactor** – Intentionally degrade one module's receipt output (add 10% noise, drop 5% of receipts). Monitor self_compression_ratio. If it drops below 0.5, alert; if recovery is slower than 10 minutes, escalate. Tests anomaly detectability of the compression metric itself.

### death_criteria

V9 should be deleted under any of these conditions:

1. **Graph divergence is not commutative**: If we prove that merge order matters for centrality, the entire receipt topology is unsound. Patch is impossible; delete and redesign around a total-order ledger.

2. **Entanglement coefficient does not converge in <100ms for any real deployment**: If query latency becomes p99 > 10s due to fixed-point solver, the system is not usable. Patch cannot recover; move to a simpler aggregation model (lossy, but fast).

3. **Counterfactual fidelity drops below 70%**: If empirical validation shows that replayed incidents match real incidents <70% of the time, the "what if" capability is unreliable. Patch cannot fix a wrong model; delete the counterfactual module and revert to forward-only tracing.

4. **Self-compression_ratio ceases to correlate with real anomalies**: Run a controlled anomaly injection campaign. If compression_ratio fails to drop when real faults are injected, the metric is cargo-cult. Patch cannot fix a useless metric; redesign health checks from scratch.

5. **Cross-domain pattern transfer success rate <50%**: If patterns learned on one domain perform <50% as well on another domain, the assumption of entanglement is wrong. Delete multi-domain entanglement and require domain-specific pattern repositories.

6. **Manual override of any paradigm occurs >3 times per quarter**: If engineers must circumvent the monad, mode predicates, or centrality to meet business needs, the constraints are unsustainable. Deletion is cleaner than patching around exceptions.

---

## How This Aligns with CLAUDEME.md

**Section 3.6 Simplicity Rule**: Each paradigm above is explainable in under 2 minutes. The one-line test for each is a whiteboard sketch. If you need more than 5 minutes, the design is invalid.

**Section 5.5 The Unfinished Manifesto**: open_questions, entropy_hooks, and death_criteria above follow the pattern. They are not finished; they are living. Any engineer can propose edits, but only proofs—never vibes—can merge.

---

## References in Codebase

These paradigms are implemented in:
- `qed.py`: Receipt monad core, graph centrality, compression metrics
- `hooks/`: Pattern definitions (all V9 modules emit R → R functions)
- `shared_anomalies.jsonl`: Entangled pattern library (no dollar_value_annual field)
- Deployment configs: Mode is a query predicate, not stored state
