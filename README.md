# consequencegraph

**Consequence-aware code knowledge graph for Python ML codebases.**  
*Rank downstream breakage by severity. Simulate changes before you make them. See cross-repo format contracts as first-class graph edges.*

🌐 **[Try the Live Demo](https://consequencegraph.onrender.com)**



---

## The problem this solves

You're working on a large ML codebase. You want to change `WeatherDataset.__getitem__` — add a new tensor to the return tuple.

You run **Find References**. It shows 12 files. You open each one. Most just mention the name in a comment. Three actually unpack the return value. One of those sites is inside a Lightning training hook, which feeds into a loss computation, which feeds into gradient accumulation. You find this out not by reading the tool output — but by reading 12 files, tracing the data manually, and keeping a mental map of what breaks.

This is the **consequence problem.** Not "what references this name" but "what breaks, specifically, and why, if I change what this returns."

The problem compounds in ML codebases:

- **Tensor shape contracts are implicit.** `training_step` expects `[B, N_grid, d_state]`. This expectation lives in the code that unpacks the batch, not the function that produces it. When the shape changes, the error surfaces at runtime — usually deep in a training loop.
- **Data flows cross abstraction boundaries.** A `Datastore` writes `.pt` files. A model in a *different repo* reads them. Changes to the writer have consequences that jump across repos in ways AST analysis cannot see.
- **Framework hooks create invisible coupling.** PyTorch Lightning's `training_step`, `validation_step`, `configure_optimizers` are called by the framework. A signature change breaks the contract silently — no call site in your code to find.
- **Architectural decisions have non-obvious blast radii.** Adding spatial weighting to the loss: does it live in `Datastore` or `metrics.py`? Either compiles. One requires changes at 3 structural points you didn't know about, the other at 7.

Standard tools give you **navigation** — find where X is used.  
consequencegraph gives you **consequence reasoning** — X changing has CRITICAL severity, here are the 36 downstream nodes ranked by how badly they break, annotated with *why* each breaks given the specific type of change you're making.

---

## What it does

consequencegraph builds a **typed, severity-scored knowledge graph** of your Python codebase by static analysis. Each node is a function, class, module, tensor contract, config key, or **cross-repo format contract**. Each edge encodes *why* the dependency exists:

| Edge type | Meaning |
|---|---|
| `produces_tensor_for` | Output shape of A feeds into B |
| `consumes_format` | B expects data in the format A produces |
| `overrides_hook` | A implements a framework lifecycle hook |
| `reads_config` | A reads a config key that B owns |
| `inherits` | A is a subclass of B |
| `calls` | A directly invokes B |
| `instantiates` | A creates an instance of B |

On top of this graph, consequencegraph provides:

**1. Blast radius analysis.** Click any node and see its full upstream/downstream dependency chain with severity scoring. Nodes are ranked by how badly they break. Edges are annotated with the reason they exist.

**2. Change simulation.** Describe a change in plain English — "add a new tensor to the return tuple", "rename this parameter", "change the hidden dimension". The tool classifies the change type and annotates each downstream node with *what specifically breaks* given that change type. Not a generic "this depends on it" — a specific "this unpacks the return value and tuple destructuring will fail."

**3. Tuple arity mismatch detection.** For return format / tensor addition changes, consequencegraph parses the AST of every `consumes_format` consumer, infers their current unpack arity, and flags nodes that will raise `ValueError: too many values to unpack` at runtime — before you run a single training step.

**4. Cross-repo format contract graph.** The wmg preset makes the invisible coupling between [weather-model-graphs](https://github.com/mllam/weather-model-graphs) and [neural-lam](https://github.com/mllam/neural-lam) visible as first-class graph nodes: `format_contract::g2m_edge_index.pt`, `format_contract::m2m_edge_index.pt`, etc. Changing `to_pyg()` in wmg now shows its full consequence in neural-lam's `load_graph` — even though no Python import connects them.

**5. Architectural reasoning.** Paste a free-form design question mentioning multiple components. Runs intersection analysis across all mentioned nodes and surfaces only the structurally load-bearing nodes for your specific decision.

---

## Why not just use Cursor / Copilot / Sourcegraph

**Cursor and Copilot** retrieve by semantic similarity. They'll surface the files you already mentioned. They won't surface `ARModel.__init__` as the place where `register_buffer` must be called, because `ARModel.__init__` isn't *semantically similar* to your query — it's *causally necessary*. Retrieval by similarity and traversal by causality are fundamentally different operations.

**Sourcegraph** gives you cross-repository reference tracking with equal weight for every caller. No concept of severity, edge type, or "this caller will break catastrophically while this other caller is unaffected."

**Find References in your IDE** is purely syntactic. It finds the name. It has no model of why the dependency exists.

The closest equivalent would be `pyan3` + `pyright` + a PyTorch Lightning expert + a custom severity scorer + a cross-repo file contract tracker + an LLM context serializer. That combination doesn't exist as a single tool. This is it.

---

## Installation

```bash
git clone https://github.com/Raj-Taware/consequence_graph.git
cd consequence_graph
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, `fastapi`, `uvicorn`, `networkx`

---

## Quickstart

**Index any Python codebase:**

```bash
python server.py --path ./your_project --port 7842
```

Open `http://localhost:7842`.

**For neural-lam only:**

```bash
python server.py --path ./neural-lam --preset neural_lam --reindex
```

**For neural-lam + weather-model-graphs (cross-repo):**

```bash
# Put both repos under a parent directory first
# e.g. gsoc_repos/neural-lam/ and gsoc_repos/weather-model-graphs/

python server.py --path ./gsoc_repos --preset wmg_neural_lam --reindex
```

This indexes both codebases together (628+ nodes, 1800+ edges) and injects the wmg format contract graph — making cross-repo consequences visible.

**Force re-index after adding a new repo:**

```bash
python server.py --path ./gsoc_repos --preset wmg_neural_lam --reindex
```

**CLI usage:**

```bash
# Index
python cli.py index ./gsoc_repos --preset wmg_neural_lam

# Impact analysis
python cli.py impact WeatherDataset.__getitem__
python cli.py impact training_step --depth 3 --human

# Pipe to LLM context
python cli.py impact ARModel.forward --llm claude

# Watch for file changes and re-index live
python cli.py watch ./gsoc_repos
```

---

## Visual interface

**Free exploration** — Force-directed graph of the full codebase. Nodes sized by in-degree. Color-coded by type. Cross-repo format contract nodes colored by mismatch risk (high / medium / low). Zoom reveals labels progressively. Click a node to fly to it.

**Focus view** — Toggles the graph into an ego-network mode. Nodes outside a 2-hop radius of your selection (or the top 7 hubs by default) fade out, making massive graphs readable.

**Blast radius** — Click any node. Upstream dependencies glow, downstream consequences highlight, intensity by severity. Sidebar shows the full impact report with edge-type annotations and tensor shape contracts.

**Change simulation ("View Consequence" panel):**
- Click a node → textarea pre-fills with structured context (type, severity, signature, tensor contracts)
- Describe your change in plain English
- Get a tiered consequence report:
  - **Will break** — must fix - must fix
  - **Likely needs updates update** — review before shipping  
  - **In scope** — broader impact area
- **Comparison Verdicts** — Ask "Should I use A or B?" to get a ranked comparison of structural cost ratios and a clear verdict on which path has fewer downstream consequences.
- **Config impact traversal** — Config keys (`config::`) are intelligently traversed upstream (who reads them) to accurately identify hyperparameter blast radii.
- **Arity warnings** highlighted in red where return format changes cause `ValueError` at runtime

**Diff view** — Click **"Diff vs main"**. Every function changed on the current branch, ranked by severity. See the blast radius of your diff before a reviewer does.

---

## Cross-repo format contracts (wmg preset)

The wmg preset makes the `.pt` / `.npz` files that wmg writes and neural-lam reads visible as first-class graph nodes:

| Contract node | Producer (wmg) | Consumer (neural-lam) | Mismatch risk |
|---|---|---|---|
| `format_contract::g2m_edge_index.pt` | `to_pyg`, `to_neural_lam` | `load_graph` | medium |
| `format_contract::m2m_edge_index.pt` | `to_pyg`, `to_neural_lam` | `load_graph` | **high** |
| `format_contract::m2m_up_*.pt` | `split_graph_by_edge_attribute` | `load_graph` | **high** |
| `format_contract::decode_mask.pt` | `export_decode_mask` | `load_decode_mask` | medium |

**Known mismatches already surfaced:**
1. `m2m_edge_index.pt` — wmg saves `Tensor(2,E)`, neural-lam's `MessagePassing` expects `List[Tensor]`
2. `m2m_up_*.pt` — wmg naming uses `m2m_up_*`, neural-lam expects `mesh_up_*` — `FileNotFoundError` at runtime
3. `from_networkx()` re-indexes all nodes to `[0, N)`, silently destroying wmg's globally-unique node labels — marked with a red ring in the graph

---

## Shape inference

Tensor shapes inferred without hardcoded presets, using a priority-ordered pipeline:

1. **jaxtyping annotations** — `Float[Tensor, "batch nodes features"]` → 95% confidence
2. **Docstring Args blocks** — `send_rep (N_send, d_h): ...` → 85% confidence
3. **Inline shape comments** — `# shape: [B, N, d_h]` → 90% confidence
4. **`nn.Linear` weight shapes** from `__init__` → 85% confidence
5. **Return annotations** — `-> tuple[Tensor, Tensor]` → arity inference

Every inferred shape carries a confidence score and source label visible in the UI.

---

## Project structure

```
consequencegraph/
├── core/
│   ├── graph.py            # Node/edge schema, KnowledgeGraph class
│   ├── indexer.py          # AST walker — builds nodes and call edges
│   ├── enricher.py         # Semantic enricher — adds typed edges
│   ├── query.py            # Blast radius, severity scoring, impact reports
│   ├── shape_inference.py  # Tensor shape extraction pipeline
│   └── watcher.py          # File watcher for live re-indexing
├── output/
│   └── llm_context.py      # Serializers for LLM context injection
├── presets/
│   ├── neural_lam.py       # neural-lam domain knowledge overrides
│   └── wmg.py              # weather-model-graphs format contract graph
├── server.py               # FastAPI server + D3.js visual interface
├── cli.py                  # Command-line interface
└── requirements.txt
```

**Preset options:**
- `neural_lam` — neural-lam only (tensor contracts, Lightning hooks, pipeline edges)
- `wmg` — weather-model-graphs only (format contract nodes, lossiness edges)
- `wmg_neural_lam` — both repos together (full cross-repo consequence graph)

---

## Known limitations

**What static analysis cannot trace:**

- Method calls through attribute-stored objects — `self.encoding_grid_mesh.forward()` where `encoding_grid_mesh` is an `InteractionNet` stored in `__init__`. The call exists at runtime; the static edge does not.
- Dynamic dispatch — `getattr(model, hook_name)()` style calls.
- Subclass method inheritance — `GraphLAM` inherits `training_step` from `ARModel` but doesn't appear in `training_step`'s downstream unless it overrides the method.

These are known gaps, not hidden ones. Every edge carries a `reason` field. The absence of an edge is a documented limitation, not a silent false negative.

---

## License

MIT
