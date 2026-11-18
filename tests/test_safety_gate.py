from evals.safety_eval import confusion, gates, load


def test_safety_set_passes_gates():
    m, _ = confusion(load())
    assert gates(m) == []
