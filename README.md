# CONSEQUENCE GRAPH

**Consequence-aware code knowledge graph for Python ML codebases.**

---

## The problem this solves

You're working on a large ML codebase. You want to change `WeatherDataset.__getitem__` — add a new tensor to the return tuple to carry spatial weight information.

You open your IDE and run **Find References**. It shows 12 files.

You open each one. Most just mention the name in a comment or import. Three of them actually unpack the return value. One of those unpacking sites is inside a Lightning training hook, which feeds into a loss computation, which feeds into gradient accumulation. You find this out not by reading the tool output — but by reading 12 files, tracing the data manually, and keeping a mental map of what breaks.

This is the **consequence problem**. Not "what references this name" but "what breaks, specifically, and why, if I change what this returns."

The problem compounds in ML codebases in particular:

- **Tensor shape contracts are implicit.** `training_step` expects a batch of shape `[B, N_grid, d_state]`. This expectation lives in the code that unpacks the batch, not in the function that produces it. There is no type system enforcing this. When the shape changes, the error surfaces at runtime — usually deep in a training loop, hours later.

- **Data flows cross abstraction boundaries.** A `Datastore` produces a graph. An `ARModel` consumes it. A `WeatherDataset` mediates between them. Changes to any one of these have consequences that jump across file and module boundaries in ways that `grep` cannot follow.

- **Framework hooks create invisible coupling.** PyTorch Lightning's `training_step`, `validation_step`, `configure_optimizers` — these are called by the framework, not by your code. A change to the signature of one of these hooks breaks the framework contract silently. No call site in your codebase to find.

- **Architectural decisions have non-obvious blast radii.** You're thinking about adding spatial weighting to the loss. Should it live in `Datastore` or `metrics.py`? Either choice compiles. Either choice runs. But one requires changes at 3 structural points you didn't know about, and the other requires changes at 7. Without a graph of the causal structure, both feel equivalent until they aren't.

Standard tools give you **navigation** — find where X is used. codegraph gives you **consequence reasoning** — X changing has CRITICAL severity, here are the 36 downstream nodes ranked by how badly they break, annotated with *why* each one breaks given the specific type of change you're making.

---

## What it does

codegraph builds a **typed, severity-scored knowledge graph** of your Python codebase by static analysis. Each node is a function, class, module, tensor contract, or config key. Each edge has a type that encodes *why* the dependency exists:

| Edge type | Meaning |
|---|---|
| `produces_tensor_for` | Output shape of A feeds into B |
| `consumes_format` | B expects data in the format A produces |
| `overrides_hook` | A implements a framework lifecycle hook |
| `reads_config` | A reads a config key that B owns |
| `inherits` | A is a subclass of B |
| `calls` | A directly invokes B |
| `instantiates` | A creates an instance of B |

On top of this graph, codegraph provides three things:

**1. Blast radius analysis.** Click any node (or query by name) and see its full upstream and downstream dependency chain, with severity scoring. Nodes are ranked by how badly they break. Edges are annotated with the reason they exist.

**2. Change simulation.** Describe a change in plain English — "add a new tensor to the return tuple", "rename this parameter", "change the hidden dimension". The tool classifies the change type and annotates each downstream node with *what specifically breaks* given that change. Not a generic "this depends on it" — a specific "this unpacks the return value and tuple destructuring will fail."

**3. Architectural reasoning.** Paste a free-form design question mentioning multiple components. The tool extracts every node you mention, computes their combined blast radii, finds the intersection — the nodes that appear in *multiple* blast radii — and surfaces only the structurally load-bearing nodes for your specific decision, with annotations explaining why each one matters.

---

## Why not just use Cursor / Copilot / Sourcegraph

**Cursor and Copilot** retrieve by semantic similarity. They embed your query, find the most similar code chunks, and let an LLM reason over them. For a design question about spatial weighting, they'll surface the files you already mentioned. They won't surface `ARModel.__init__` as the place where `register_buffer` must be called, because `ARModel.__init__` isn't semantically similar to your query — it's *causally necessary* for your change. Retrieval by similarity and traversal by causality are fundamentally different operations.

**Sourcegraph** gives you cross-repository reference tracking. It answers "who calls this function" with equal weight for every caller. It has no concept of severity, no concept of edge type, no concept of "this caller will break catastrophically while this other caller will be unaffected." It shows you all references. codegraph shows you ranked consequences.

**Find References in your IDE** is purely syntactic. It finds the name. It has no model of why the dependency exists or what kind of change would break it.

The closest equivalent to codegraph would be `pyan3` (call graphs) + `pyright` (type inference) + a PyTorch Lightning expert + a custom severity scorer + an LLM context serializer. That combination doesn't exist as a single tool. This is it.

---

## Installation

```bash
git clone https://github.com/yourname/codegraph
cd codegraph
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, `fastapi`, `uvicorn`, `networkx`

---

## Quickstart

**Index a codebase and open the visual interface:**

```bash
python server.py --path ./your_project --port 7842
```

Open `http://localhost:7842`.

**For neural-lam specifically:**

```bash
python server.py --path ./neural_lam --preset neural_lam --reindex
```

**CLI usage:**

```bash
# Index
python cli.py index ./neural_lam --preset neural_lam

# Impact analysis
python cli.py impact WeatherDataset.__getitem__
python cli.py impact training_step --depth 3 --human

# Pipe to LLM context
python cli.py impact ARModel.forward --llm claude

# Watch for file changes and re-index live
python cli.py watch ./neural_lam
```

---

## Visual interface

The browser UI has three layers:

**Free exploration** — Force-directed graph of the full codebase. Nodes sized by in-degree (how many things depend on them). Color-coded by type. Zoom reveals labels progressively — high-degree hub nodes always visible, detail emerges as you zoom in. Click a node and the camera flies to it.

**Blast radius** — Click any node. Upstream dependencies glow blue, downstream consequences glow red, intensity by severity. Everything else fades. Sidebar shows the full impact report with edge-type annotations and tensor shape contracts where inferred.

**Design reasoning** — Two panels at the bottom of the sidebar:

- *Simulate change* — Select a node, describe a change in plain English, get consequence predictions annotated per affected node for that specific change type.
- *Reason about design* — Paste a multi-node architectural question. Runs intersection analysis across all mentioned nodes and returns only the structurally load-bearing nodes for your specific decision.

---

## Shape inference

codegraph infers tensor shapes without hardcoded presets, using a priority-ordered pipeline:

1. **jaxtyping annotations** — `Float[Tensor, "batch nodes features"]` → 95% confidence
2. **Docstring Args blocks** — `send_rep (N_send, d_h): ...` → 85% confidence
3. **Inline shape comments** — `# shape: [B, N, d_h]` → 90% confidence
4. **`nn.Linear` weight shapes** — `nn.Linear(hidden_dim, output_dim)` in `__init__` → 85% confidence
5. **Return annotations** — `-> tuple[Tensor, Tensor]` → arity inference

Every inferred shape carries a confidence score and source label visible in the UI.

---

## Domain presets

Presets add expert knowledge about ML frameworks as a queryable graph layer on top of the AST-inferred graph. The neural-lam preset adds tensor shape contracts, pipeline edges between dataset/model/training, and Lightning hook detection.

Presets are override files — auto-inference runs first, preset corrections apply second. To add a preset for your own codebase, see `presets/neural_lam.py`.

---

## Git diff integration

Click **"Diff vs main"** in the UI to see every function changed on the current branch, ranked by severity of downstream impact. Useful before merging: see exactly which downstream nodes the diff touches before a reviewer has to find out the hard way.

---

## Project structure

```
codegraph/
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
│   └── neural_lam.py       # neural-lam domain knowledge overrides
├── server.py               # FastAPI server + D3.js visual interface
├── cli.py                  # Command-line interface
└── requirements.txt
```

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
