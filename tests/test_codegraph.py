"""
Tests for codegraph.
Run with: pytest tests/ -v
"""
import sys
import os
import textwrap
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.graph import KnowledgeGraph, NodeData, EdgeData, NodeType, EdgeType
from core.indexer import Indexer
from core.enricher import Enricher
from core.query import QueryEngine
from output.llm_context import to_json, format_impact_as_context, format_for_llm


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _write_files(tmpdir, files: dict[str, str]) -> str:
    for rel_path, content in files.items():
        full = os.path.join(tmpdir, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(textwrap.dedent(content))
    return tmpdir


def _build_graph(files: dict[str, str]) -> tuple[KnowledgeGraph, str]:
    tmpdir = tempfile.mkdtemp()
    _write_files(tmpdir, files)
    graph = KnowledgeGraph()
    Indexer(graph, tmpdir).index_path(tmpdir)
    Enricher(graph).run()
    return graph, tmpdir


# ── Graph schema tests ────────────────────────────────────────────────────────

class TestKnowledgeGraph:
    def test_add_and_get_node(self):
        g = KnowledgeGraph()
        node = NodeData(
            node_id="mymod.MyClass",
            node_type=NodeType.CLASS,
            name="MyClass",
            module="mymod",
            file_path="mymod.py",
        )
        g.add_node(node)
        assert g.has_node("mymod.MyClass")
        data = g.get_node("mymod.MyClass")
        assert data["name"] == "MyClass"

    def test_add_edge_only_if_nodes_exist(self):
        g = KnowledgeGraph()
        n1 = NodeData("a", NodeType.FUNCTION, "a", "m", "f.py")
        n2 = NodeData("b", NodeType.FUNCTION, "b", "m", "f.py")
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge("a", "b", EdgeData(EdgeType.CALLS, "test"))
        assert g.edge_count() == 1
        # Edge to non-existent node should be silently dropped
        g.add_edge("a", "nonexistent", EdgeData(EdgeType.CALLS, "test"))
        assert g.edge_count() == 1

    def test_resolve_node_id_fuzzy(self):
        g = KnowledgeGraph()
        g.add_node(NodeData("pkg.mod.MyClass.my_method", NodeType.FUNCTION, "my_method", "pkg.mod", "f.py"))
        # exact
        assert g.resolve_node_id("pkg.mod.MyClass.my_method") == ["pkg.mod.MyClass.my_method"]
        # suffix
        assert "pkg.mod.MyClass.my_method" in g.resolve_node_id("MyClass.my_method")
        # substring
        assert "pkg.mod.MyClass.my_method" in g.resolve_node_id("my_method")

    def test_remove_nodes_for_file(self):
        g = KnowledgeGraph()
        g.add_node(NodeData("a", NodeType.FUNCTION, "a", "m", "file_a.py"))
        g.add_node(NodeData("b", NodeType.FUNCTION, "b", "m", "file_b.py"))
        g.remove_nodes_for_file("file_a.py")
        assert not g.has_node("a")
        assert g.has_node("b")

    def test_save_load(self, tmp_path):
        g = KnowledgeGraph()
        g.add_node(NodeData("x.y", NodeType.MODULE, "y", "x", "x/y.py"))
        cache = str(tmp_path / "cache.json")
        g.save(cache)
        g2 = KnowledgeGraph()
        assert g2.load(cache)
        assert g2.has_node("x.y")


# ── Indexer tests ─────────────────────────────────────────────────────────────

class TestIndexer:
    def test_indexes_module_class_function(self):
        graph, _ = _build_graph({
            "pkg/mod.py": """
                class Foo:
                    def bar(self):
                        pass
                def top_level():
                    pass
            """
        })
        assert graph.has_node("pkg.mod")
        # class and function should be indexed
        nodes = {nid for nid in graph.g.nodes}
        assert any("Foo" in n for n in nodes)
        assert any("bar" in n for n in nodes)
        assert any("top_level" in n for n in nodes)

    def test_lightning_hook_detection(self):
        graph, _ = _build_graph({
            "model.py": """
                import pytorch_lightning as pl
                class MyModel(pl.LightningModule):
                    def training_step(self, batch, idx):
                        pass
                    def forward(self, x):
                        pass
            """
        })
        # training_step should be flagged as lightning hook
        training_nodes = [
            (nid, d) for nid, d in graph.g.nodes(data=True)
            if d.get("name") == "training_step"
        ]
        assert len(training_nodes) > 0
        assert training_nodes[0][1]["is_lightning_hook"] is True

    def test_config_key_extraction(self):
        graph, _ = _build_graph({
            "model.py": """
                class M:
                    def __init__(self, hparams):
                        self.hidden = self.hparams.hidden_dim
                        val = hparams['lr']
            """
        })
        cfg_nodes = [nid for nid in graph.g.nodes if nid.startswith("config::")]
        assert "config::hidden_dim" in cfg_nodes or "config::lr" in cfg_nodes

    def test_inheritance_edge(self):
        graph, _ = _build_graph({
            "base.py": """
                class Base:
                    pass
            """,
            "child.py": """
                from base import Base
                class Child(Base):
                    pass
            """
        })
        # Should have an inherits edge from Child to Base
        inherits_edges = [
            (u, v, d) for u, v, d in graph.g.edges(data=True)
            if d.get("edge_type") == EdgeType.INHERITS.value
        ]
        assert len(inherits_edges) > 0

    def test_incremental_reindex(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "mod.py")
        with open(path, "w") as f:
            f.write("def foo(): pass\n")
        graph = KnowledgeGraph()
        Indexer(graph, tmpdir).index_path(tmpdir)
        assert any("foo" in n for n in graph.g.nodes)
        # Rewrite file with different function
        with open(path, "w") as f:
            f.write("def bar(): pass\n")
        Indexer(graph, tmpdir).reindex_file(path)
        # foo should be gone, bar should be present
        assert not any(n.endswith(".foo") for n in graph.g.nodes)
        assert any("bar" in n for n in graph.g.nodes)


# ── Enricher tests ────────────────────────────────────────────────────────────

class TestEnricher:
    def test_dataset_to_training_step_edge(self):
        graph, _ = _build_graph({
            "dataset.py": """
                class WeatherDataset:
                    def __getitem__(self, idx):
                        return idx
            """,
            "model.py": """
                class Model:
                    def training_step(self, batch, idx):
                        pass
                    def validation_step(self, batch, idx):
                        pass
            """
        })
        edges = [
            (u, v, d) for u, v, d in graph.g.edges(data=True)
            if d.get("edge_type") == EdgeType.CONSUMES_FORMAT.value
        ]
        assert len(edges) > 0

    def test_naming_risk_flags_docstring_references(self):
        graph, _ = _build_graph({
            "mod.py": """
                def helper():
                    pass
                def main():
                    \"\"\"Calls helper to do work.\"\"\"
                    pass
            """
        })
        named_ref_edges = [
            (u, v, d) for u, v, d in graph.g.edges(data=True)
            if d.get("edge_type") == EdgeType.NAMED_REFERENCE.value
        ]
        assert len(named_ref_edges) > 0


# ── Query engine tests ────────────────────────────────────────────────────────

class TestQueryEngine:
    def test_impact_returns_valid_structure(self):
        graph, _ = _build_graph({
            "pkg/a.py": """
                def foo():
                    pass
                def bar():
                    foo()
            """
        })
        engine = QueryEngine(graph)
        result = engine.impact("foo")
        assert "target" in result
        assert "severity" in result
        assert "blast_radius" in result
        assert "upstream" in result["blast_radius"]
        assert "downstream" in result["blast_radius"]

    def test_impact_not_found(self):
        graph = KnowledgeGraph()
        engine = QueryEngine(graph)
        result = engine.impact("nonexistent_function_xyz")
        assert "error" in result

    def test_impact_ambiguous(self):
        graph, _ = _build_graph({
            "a.py": "def process(): pass",
            "b.py": "def process(): pass",
        })
        engine = QueryEngine(graph)
        result = engine.impact("process")
        # Should either resolve both or return ambiguous
        assert "target" in result or "ambiguous" in result

    def test_manual_depth_respected(self):
        graph, _ = _build_graph({
            "pkg/a.py": """
                def func_alpha(): pass
                def func_beta(): func_alpha()
                def func_gamma(): func_beta()
                def func_delta(): func_gamma()
            """
        })
        engine = QueryEngine(graph)
        r1 = engine.impact("func_alpha", depth=1)
        r3 = engine.impact("func_alpha", depth=3)
        # func_alpha should be unique enough
        if "ambiguous" in r1:
            return  # skip if still ambiguous
        assert r1["depth_used"] == 1
        assert r1["depth_mode"] == "manual"
        assert r3["depth_used"] == 3

    def test_auto_depth_scales_with_severity(self):
        """High-severity nodes (lightning hooks) should get larger auto depth."""
        graph, _ = _build_graph({
            "model.py": """
                import pytorch_lightning as pl
                class M(pl.LightningModule):
                    def training_step(self, batch, idx):
                        pass
                def simple_util():
                    pass
            """
        })
        engine = QueryEngine(graph)
        hook_result = engine.impact("training_step")
        leaf_result = engine.impact("simple_util")
        assert hook_result["depth_used"] >= leaf_result["depth_used"]

    def test_severity_label_range(self):
        graph, _ = _build_graph({"mod.py": "def x(): pass"})
        engine = QueryEngine(graph)
        result = engine.impact("x")
        assert result["severity"] in ("low", "medium", "high", "critical")


# ── Output formatter tests ────────────────────────────────────────────────────

class TestOutputFormatters:
    def _make_impact(self):
        graph, _ = _build_graph({
            "pkg/model.py": """
                import pytorch_lightning as pl
                class Model(pl.LightningModule):
                    def forward(self, x):
                        # shape: [B, N, d]
                        return x
                    def training_step(self, batch, idx):
                        pass
            """
        })
        engine = QueryEngine(graph)
        return engine.impact("forward")

    def test_json_output_is_valid(self):
        import json
        impact = self._make_impact()
        out = to_json(impact)
        parsed = json.loads(out)
        assert "target" in parsed

    def test_human_output_contains_key_sections(self):
        impact = self._make_impact()
        out = format_impact_as_context(impact)
        assert "CODEGRAPH IMPACT CONTEXT" in out
        assert "SUMMARY" in out
        assert "UPSTREAM" in out or "DOWNSTREAM" in out

    def test_llm_adapters_all_work(self):
        impact = self._make_impact()
        for adapter in ["claude", "openai", "ollama", "generic"]:
            out = format_for_llm(impact, llm=adapter)
            assert len(out) > 100

    def test_claude_adapter_wraps_in_context_tags(self):
        impact = self._make_impact()
        out = format_for_llm(impact, llm="claude")
        assert "<context>" in out
        assert "</context>" in out

    def test_error_impact_formats_cleanly(self):
        graph = KnowledgeGraph()
        engine = QueryEngine(graph)
        result = engine.impact("nothing")
        out = format_impact_as_context(result)
        assert "ERROR" in out or "not found" in out.lower()
