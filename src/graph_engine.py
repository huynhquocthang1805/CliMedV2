"""
graph_engine.py — CliMedV2
==========================

Knowledge Graph cho SXHD: load YAML curated → NetworkX → reasoning queries.

Pipeline:
    graph_seed.yaml ──► KnowledgeGraph.load
                                │
                                ▼
                  patient context (rule_output, ocr_records, symptoms)
                                │
                                ▼
                         activate(context) → list[seed_nodes]
                                │
                                ▼
                         traverse(seeds, hops=2) → relations
                                │
                                ▼
                  format_for_llm(facts) → text block cho Qwen

Khác với GraphRAG general-purpose:
  • Không có entity extraction LLM (đã curated trong YAML).
  • Tập trung vào REASONING multi-hop có giải thích, không phải retrieval.
  • Mỗi fact đi kèm path để bác sĩ verify được vì sao bot suy luận thế.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


# ===========================================================================
# Path setup
# ===========================================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_YAML = BASE_DIR / "data" / "knowledge" / "graph_seed.yaml"


# ===========================================================================
# Data classes
# ===========================================================================

@dataclass
class GraphFact:
    """1 fact derived từ graph traversal — kèm path để giải thích."""
    summary: str               # human-readable, vd "PLT thấp gợi ý SXHD cảnh báo"
    path: List[str]            # node ids theo thứ tự
    relation_chain: List[str]  # các edge relations theo thứ tự
    score: float = 1.0         # trọng số (1.0 default, có thể tăng cho high-impact)

    def to_text(self, graph: "KnowledgeGraph") -> str:
        """Render fact thành 1 dòng text với citation."""
        labels = [graph.label_of(nid) for nid in self.path]
        chain = " → ".join(labels)
        rel_str = " / ".join(self.relation_chain) if self.relation_chain else ""
        return f"• {self.summary}  [{chain} | rel: {rel_str}]"


# ===========================================================================
# KnowledgeGraph wrapper
# ===========================================================================

# Mapping relation → tiếng Việt cho explanation
_RELATION_VI = {
    "supports":              "ủng hộ",
    "warning_of":            "là dấu hiệu cảnh báo của",
    "contraindicated_for":   "chống chỉ định trong",
    "treats":                "xử trí cho",
}


class KnowledgeGraph:
    """
    Wrapper quanh nx.DiGraph với:
      • load_yaml: load curated graph
      • label_of, type_of, get_node, neighbors
      • activate: chọn các node "kích hoạt" từ patient context
      • traverse: BFS multi-hop, trả về facts có explain path
    """

    def __init__(self, graph: Optional[nx.DiGraph] = None):
        self.G: nx.DiGraph = graph if graph is not None else nx.DiGraph()

    # ----- Load / build -----

    @classmethod
    def load_yaml(cls, path: Path = DEFAULT_GRAPH_YAML) -> "KnowledgeGraph":
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML chưa cài. Chạy: pip install pyyaml"
            ) from e

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Graph seed không tồn tại: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        G = nx.DiGraph()

        # Add nodes
        for n in data.get("nodes", []):
            nid = n.pop("id")
            G.add_node(nid, **n)

        # Add edges
        for e in data.get("edges", []):
            src = e.pop("source")
            tgt = e.pop("target")
            G.add_edge(src, tgt, **e)

        logger.info("Loaded graph from %s: %d nodes, %d edges",
                    path.name, G.number_of_nodes(), G.number_of_edges())
        return cls(G)

    # ----- Lookup helpers -----

    def label_of(self, nid: str) -> str:
        if nid not in self.G.nodes:
            return nid
        return self.G.nodes[nid].get("label", nid)

    def type_of(self, nid: str) -> str:
        if nid not in self.G.nodes:
            return "unknown"
        return self.G.nodes[nid].get("type", "unknown")

    def nodes_by_type(self, ntype: str) -> List[str]:
        return [n for n, d in self.G.nodes(data=True)
                if d.get("type") == ntype]

    def get_attrs(self, nid: str) -> Dict[str, Any]:
        return dict(self.G.nodes.get(nid, {}))

    # ----- Activation: từ context bệnh nhân → các node kích hoạt -----

    def activate_from_context(
        self,
        symptoms: Optional[Dict[str, bool]] = None,
        lab_records: Optional[pd.DataFrame] = None,
        rule_severity: Optional[str] = None,
    ) -> List[str]:
        """
        Quyết định các node nào được "kích hoạt" cho 1 bệnh nhân cụ thể.

        Args:
            symptoms: dict {key: True/False} từ clinical_input.
                      Key map sang node id qua _SYMPTOM_TO_NODE.
            lab_records: DataFrame OCR sau parse_exam_text_to_rows.
                         Có cột exam_name_normalized, value, range_flag.
            rule_severity: text severity từ rule_engine (vd "SXHD có dấu hiệu cảnh báo").

        Returns:
            List node ids đã activate (có trong graph).
        """
        activated: Set[str] = set()

        # 1. Symptoms → node
        if symptoms:
            sym_map = self._symptom_key_to_node()
            for key, on in symptoms.items():
                if on and key in sym_map:
                    nid = sym_map[key]
                    if nid in self.G.nodes:
                        activated.add(nid)

        # 2. Lab records → node (qua threshold check)
        if lab_records is not None and not lab_records.empty:
            activated.update(self._activate_from_labs(lab_records))

        # 3. Rule severity → severity node
        if rule_severity:
            sev_node = self._severity_text_to_node(rule_severity)
            if sev_node:
                activated.add(sev_node)

        logger.info("Activated %d nodes: %s",
                    len(activated), sorted(activated)[:10])
        return sorted(activated)

    @staticmethod
    def _symptom_key_to_node() -> Dict[str, str]:
        """
        Map từ clinical_input keys (Streamlit form) → node id.
        Mở rộng map này khi thêm symptom mới vào form.
        """
        return {
            "fever_now":         "sot_cao",
            "abdominal_pain":    "dau_bung_nhieu",
            "vomiting":          "non_nhieu",
            "vomiting_many":     "non_nhieu",
            "mucosal_bleeding":  "chay_mau_niem_mac",
            "oliguria":          "tieu_it",
            "lethargy":          "lu_du_vat_va",
            "cold_extremities":  "tay_chan_lanh",
            "myalgia":           "dau_co_khop",
            "retro_orbital_pain": "dau_hoc_mat",
            "dyspnea":           "tran_dich",
            # alias key để tương thích nhiều form
            "petechia":          "ban_xuat_huyet",
            "headache":          "dau_dau",
        }

    def _activate_from_labs(self, lab_df: pd.DataFrame) -> Set[str]:
        """
        Activate node từ lab values. Logic dựa vào tên chỉ số + range_flag.
        """
        activated: Set[str] = set()
        if "exam_name_normalized" not in lab_df.columns:
            return activated

        for _, row in lab_df.iterrows():
            name = str(row.get("exam_name_normalized", "")).upper()
            flag = str(row.get("range_flag", "")).lower()
            value = row.get("value")
            try:
                value = float(value) if value is not None else None
            except (ValueError, TypeError):
                value = None

            # PLT giảm
            if name == "PLT" and (flag == "low" or
                                  (value is not None and value < 100)):
                activated.add("tieu_cau_thap")

            # HCT cao
            if name == "HCT" and flag == "high":
                activated.add("hct_cao")

            # AST/ALT cao
            if name in ("AST", "ALT") and (flag == "high" or
                                            (value is not None and value > 50)):
                activated.add("ast_alt_cao")

            # WBC giảm
            if name == "WBC" and flag == "low":
                activated.add("bach_cau_thap")

            # HFLC cao (>10%)
            if "HFLC" in name and value is not None and value > 10:
                activated.add("hflc_cao")

        return activated

    @staticmethod
    def _severity_text_to_node(text: str) -> Optional[str]:
        """Map free-text severity → node id."""
        t = text.lower()
        if "nặng" in t or "soc" in t or "sốc" in t:
            return "sxhd_nang"
        if "cảnh báo" in t or "canh bao" in t or "warning" in t:
            return "sxhd_canh_bao"
        if "thường" in t or "thuong" in t or "common" in t:
            return "sxhd_thuong"
        return None

    # ----- Traversal: multi-hop reasoning với explanation -----

    def traverse(
        self,
        seed_nodes: List[str],
        max_hops: int = 2,
        relation_filter: Optional[List[str]] = None,
    ) -> List[GraphFact]:
        """
        BFS từ seed nodes, mỗi step ghi lại path + relation chain.

        Trả về list facts.
        relation_filter: nếu set, chỉ đi theo các relation này.
        """
        if not seed_nodes:
            return []

        facts: List[GraphFact] = []
        visited_paths: Set[Tuple[str, ...]] = set()

        # BFS với queue (path, relations)
        queue: List[Tuple[List[str], List[str]]] = [
            ([s], []) for s in seed_nodes if s in self.G.nodes
        ]

        while queue:
            path, rels = queue.pop(0)
            current = path[-1]

            # Đã đến max hops
            if len(rels) >= max_hops:
                continue

            for nbr in self.G.successors(current):
                if nbr in path:  # tránh cycle
                    continue
                edge_data = self.G.get_edge_data(current, nbr)
                rel = edge_data.get("relation", "related_to")
                if relation_filter and rel not in relation_filter:
                    continue

                new_path = path + [nbr]
                new_rels = rels + [rel]
                key = tuple(new_path)
                if key in visited_paths:
                    continue
                visited_paths.add(key)

                # Sinh fact
                summary = self._summarize_path(new_path, new_rels, edge_data)
                weight = float(edge_data.get("weight", 1.0))
                facts.append(GraphFact(
                    summary=summary,
                    path=new_path,
                    relation_chain=new_rels,
                    score=weight * (1.0 / len(new_rels)),  # gần seed → score cao
                ))

                # Tiếp tục BFS
                queue.append((new_path, new_rels))

        # Sort theo score giảm dần
        facts.sort(key=lambda f: -f.score)
        return facts

    def _summarize_path(self, path: List[str], rels: List[str],
                        last_edge: Dict[str, Any]) -> str:
        """Viết summary câu cho 1 fact."""
        if len(path) < 2:
            return self.label_of(path[0])

        src_label = self.label_of(path[0])
        tgt_label = self.label_of(path[-1])
        last_rel = rels[-1] if rels else "related_to"
        rel_vi = _RELATION_VI.get(last_rel, last_rel)

        if len(path) == 2:
            # 1-hop: trực tiếp
            return f"{src_label} {rel_vi} {tgt_label}"
        # multi-hop: nói chain
        chain = " → ".join(self.label_of(n) for n in path)
        return f"{src_label} → ... → {tgt_label} (qua {len(rels)} bước: {chain})"

    # ----- Aggregate facts theo target severity (cho rule cross-check) -----

    def aggregate_severity_evidence(
        self, activated: List[str], max_hops: int = 1,
    ) -> Dict[str, List[GraphFact]]:
        """
        Group facts theo severity node đến đích.
        Hữu ích để show: "Có X bằng chứng cho SXHD cảnh báo, Y cho SXHD nặng".
        """
        all_facts = self.traverse(activated, max_hops=max_hops)
        by_severity: Dict[str, List[GraphFact]] = {}
        for f in all_facts:
            tgt = f.path[-1]
            if self.type_of(tgt) == "severity":
                by_severity.setdefault(tgt, []).append(f)
        return by_severity


# ===========================================================================
# Public API: format kết quả cho LLM context
# ===========================================================================

def format_graph_facts_for_llm(
    graph: KnowledgeGraph,
    activated: List[str],
    facts: List[GraphFact],
    max_facts: int = 12,
) -> str:
    """
    Format toàn bộ thành 1 text block sẵn để inject vào LLM prompt.
    """
    if not activated and not facts:
        return ""

    lines = ["## QUAN HỆ TRI THỨC (từ knowledge graph SXHD)"]

    if activated:
        labels = [graph.label_of(nid) for nid in activated]
        lines.append(f"\n**Nodes kích hoạt từ ngữ cảnh BN ({len(activated)}):** "
                     + ", ".join(labels))

    if facts:
        lines.append(f"\n**Suy luận từ graph (top {min(max_facts, len(facts))}):**")
        for f in facts[:max_facts]:
            lines.append(f.to_text(graph))

    # Group theo severity
    by_sev = {}
    for f in facts[:max_facts]:
        tgt = f.path[-1]
        if graph.type_of(tgt) == "severity":
            by_sev.setdefault(graph.label_of(tgt), 0)
            by_sev[graph.label_of(tgt)] += 1
    if by_sev:
        lines.append("\n**Tóm tắt số bằng chứng theo severity:**")
        for sev, n in sorted(by_sev.items(), key=lambda x: -x[1]):
            lines.append(f"  - {sev}: {n} bằng chứng")

    return "\n".join(lines)


# ===========================================================================
# Singleton + convenience
# ===========================================================================

_GRAPH_SINGLETON: Optional[KnowledgeGraph] = None


def get_graph(force_reload: bool = False) -> KnowledgeGraph:
    """Singleton accessor — load 1 lần per process."""
    global _GRAPH_SINGLETON
    if _GRAPH_SINGLETON is None or force_reload:
        _GRAPH_SINGLETON = KnowledgeGraph.load_yaml()
    return _GRAPH_SINGLETON


def graph_facts_for_chat(
    symptoms: Optional[Dict[str, bool]] = None,
    lab_records: Optional[pd.DataFrame] = None,
    rule_severity: Optional[str] = None,
    max_hops: int = 2,
    max_facts: int = 12,
) -> str:
    """
    Convenience cho chatbot_engine: lấy text block đã format từ context.
    Trả về "" nếu không có gì để inject (graph chưa load / không activate gì).
    """
    try:
        g = get_graph()
    except Exception as e:
        logger.warning("Graph load failed: %s", e)
        return ""

    activated = g.activate_from_context(
        symptoms=symptoms,
        lab_records=lab_records,
        rule_severity=rule_severity,
    )
    if not activated:
        return ""

    facts = g.traverse(activated, max_hops=max_hops)
    return format_graph_facts_for_llm(g, activated, facts, max_facts=max_facts)


# ===========================================================================
# CLI test
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    g = get_graph()
    print(f"\n=== GRAPH STATS ===")
    print(f"Nodes: {g.G.number_of_nodes()}")
    print(f"Edges: {g.G.number_of_edges()}")
    print(f"By type:")
    from collections import Counter
    type_counts = Counter(g.type_of(n) for n in g.G.nodes)
    for t, n in type_counts.most_common():
        print(f"  {t}: {n}")

    # Demo activate + traverse
    print(f"\n=== DEMO: BN có PLT thấp + đau bụng nhiều ===")
    fake_symptoms = {"abdominal_pain": True, "vomiting_many": True,
                     "fever_now": True}
    fake_labs = pd.DataFrame([
        {"exam_name_normalized": "PLT", "value": 50, "range_flag": "low"},
        {"exam_name_normalized": "AST", "value": 200, "range_flag": "high"},
    ])
    text = graph_facts_for_chat(
        symptoms=fake_symptoms, lab_records=fake_labs,
        rule_severity="SXHD có dấu hiệu cảnh báo",
        max_hops=2,
    )
    print(text)