from app.data import load_medquad

XML = """<?xml version="1.0" encoding="UTF-8"?>
<Document id="0000804" source="GHR" url="https://ghr.nlm.nih.gov/condition/pkd">
<Focus>polycystic kidney disease</Focus>
<QAPairs>
  <QAPair pid="1"><Question qid="1" qtype="information">What is PKD ?</Question>
    <Answer>PKD is a   kidney disorder.</Answer></QAPair>
  <QAPair pid="2"><Question qid="2" qtype="causes">What causes PKD ?</Question>
    <Answer></Answer></QAPair>
</QAPairs></Document>"""


def test_parses_chunks_and_skips_excluded_subsets(tmp_path):
    (tmp_path / "3_GHR_QA").mkdir()
    (tmp_path / "3_GHR_QA" / "0000804.xml").write_text(XML)
    (tmp_path / "10_MPlus_ADAM_QA").mkdir()
    (tmp_path / "10_MPlus_ADAM_QA" / "x.xml").write_text(XML.replace("GHR", "ADAM"))

    chunks = load_medquad(tmp_path)

    assert [c.id for c in chunks] == ["GHR_0000804_Sec1"]  # empty answer skipped, ADAM excluded
    c = chunks[0]
    assert c.answer == "PKD is a kidney disorder."
    assert c.url == "https://ghr.nlm.nih.gov/condition/pkd"
    assert "What is PKD" not in c.text  # the question is never indexed
