"""Load MedQuAD XML into chunks.

One chunk = one QA pair's answer. MedQuAD documents are already split into
question-specific sections ("What causes X?", "How is X treated?"), and the
LiveQA relevance judgments are at exactly that level, so a QA pair is the
natural chunk.

Chunk id format matches the official answer ids, e.g. GHR_0000804_Sec2:
  source = GHR, Document id = 0000804, QAPair pid = 2.

The question text is NOT indexed. It is kept only as eval data, so the
MedQuAD eval measures retrieval for questions the index has never seen.
"""
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

# MedlinePlus A.D.A.M., Drugs and Herbs had their answers removed from the
# public release for copyright reasons, so they are excluded.
EXCLUDED_SUBSETS = {"10_MPlus_ADAM_QA", "11_MPlusDrugs_QA", "12_MPlusHerbsSupplements_QA"}
EXCLUDED_SOURCES = {"ADAM", "MPlusDrugs", "MPlusHerbsSuppls"}  # prefixes used in qrels


@dataclass
class Chunk:
    id: str        # GHR_0000804_Sec2
    source: str    # GHR
    url: str       # NIH page the answer comes from (used for citations)
    focus: str     # topic, e.g. "polycystic kidney disease"
    question: str  # original question (eval only, never indexed)
    qtype: str     # information / causes / treatment / ...
    answer: str

    @property
    def text(self) -> str:
        """What gets embedded and shown to the LLM."""
        return f"{self.focus}\n{self.answer}"


def parse_file(path: Path) -> list[Chunk]:
    root = ET.parse(path).getroot()
    source, doc_id, url = root.get("source"), root.get("id"), root.get("url")
    focus = (root.findtext("Focus") or "").strip()
    chunks = []
    for qa in root.iter("QAPair"):
        answer = " ".join((qa.findtext("Answer") or "").split())
        if not answer:
            continue
        q = qa.find("Question")
        chunks.append(Chunk(
            id=f"{source}_{doc_id}_Sec{qa.get('pid')}",
            source=source, url=url, focus=focus,
            question=(q.text or "").strip() if q is not None else "",
            qtype=q.get("qtype", "") if q is not None else "",
            answer=answer,
        ))
    return chunks


def load_medquad(root: str | Path) -> list[Chunk]:
    chunks, seen = [], {}
    for subset in sorted(Path(root).iterdir()):
        if not subset.is_dir() or subset.name in EXCLUDED_SUBSETS or subset.name.startswith("."):
            continue
        for f in sorted(subset.glob("*.xml")):
            for c in parse_file(f):
                # A few CancerGov files repeat pids for different answers;
                # keep them all but make the id unique.
                seen[c.id] = seen.get(c.id, 0) + 1
                if seen[c.id] > 1:
                    c.id = f"{c.id}-{seen[c.id]}"
                chunks.append(c)
    return chunks
