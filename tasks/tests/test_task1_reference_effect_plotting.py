"""Check the paper reference-effect figure's labels and geometry."""

import pytest

from wmt26_mteval_task1.reference_effect_plotting import (
    ReferenceEffectRow,
    plot_reference_effects,
    short_direction,
)


@pytest.mark.parametrize(("direction", "expected"), [
    ("ces_Latn-deu_Latn", "ces-deu"),
    ("eng_Latn-ekk_Latn", "eng-ekk"),
    ("eng_Latn-sme_Latn", "eng-sme"),
    ("zho_Hans-jpn_Jpan", "zho-jpn"),
    ("eng_Latn-zho_Hans", "eng-zho_CN"),
    ("eng_Latn-zho_Hant_TW", "eng-zho_TW"),
    ("eng-zho_CN", "eng-zho_CN"),
    ("eng-zho_TW", "eng-zho_TW"),
])
def test_project_direction_labels(direction, expected):
    """Check script-tagged input yields project codes without losing Chinese variants."""

    assert short_direction(direction) == expected


@pytest.mark.parametrize("count", [7, 8])
def test_column_geometry_and_text_fit(tmp_path, monkeypatch, count):
    """Check fixed grid boundaries and unclipped labels for both row counts."""

    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    captured = []
    close = plt.close
    monkeypatch.setattr(plt, "close", lambda figure: captured.append(figure))
    rows = [ReferenceEffectRow("eng_Latn-arz_Arab", "Post-edited → Pseudo",
                              "cESA 80.7 → 79.8", (-1) ** index * 0.9)
            for index in range(count)]
    paths = plot_reference_effects(tmp_path / "paper", rows, plt,
                                   definition="Reference A → B; ΔF1 = B − A")
    figure = captured[-1]
    try:
        assert figure.get_figwidth() == 3.5
        assert figure.get_figheight() == pytest.approx(0.35 * count + 0.72)
        directions, metadata, axis = figure.axes
        assert axis.get_position().x0 == pytest.approx(0.62)
        assert axis.get_position().x1 == pytest.approx(0.970)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        texts = (figure.texts + directions.texts + metadata.texts + axis.texts
                 + axis.get_xticklabels() + [axis.xaxis.label])
        for text in texts:
            bounds = text.get_window_extent(renderer)
            assert bounds.x0 >= 0
            assert bounds.y0 >= 0
            assert bounds.x1 <= figure.bbox.width
            assert bounds.y1 <= figure.bbox.height
        assert all(text.get_window_extent(renderer).x1 < axis.bbox.x0
                   for text in metadata.texts)
        assert all(text.get_window_extent(renderer).x1 < metadata.bbox.x0
                   for text in directions.texts)
        for badge in metadata.artists:
            bounds = badge.get_window_extent(renderer)
            assert bounds.x0 >= metadata.bbox.x0
            assert bounds.x1 < axis.bbox.x0
            assert (bounds.x0 + bounds.x1) / 2 == pytest.approx(
                (metadata.bbox.x0 + metadata.bbox.x1) / 2)
        assert all(path.stat().st_size > 0 for path in paths)
    finally:
        close(figure)
