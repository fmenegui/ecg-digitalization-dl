import numpy as np

from ecgdl.constants import LEAD_INDEX, LEADS
from ecgdl.layout.assemble import LayoutConfig, NameDetection, assemble_layout, layout_from_rows

PPMM = 2.0


def make_page(rows, rhythm=(), seg_mm=62.5, row_mm=30.0, x0=15.0, name_dy=-8.0, drop=()):
    """Ink map with one flat-ish trace per row and name detections placed like a printer does."""
    n_rows = len(rows) + len(rhythm)
    cols = len(rows[0])
    h, w = int((n_rows * row_mm + 30) * PPMM), int((cols * seg_mm + 2 * x0) * PPMM)
    ink = np.zeros((h, w), np.float32)
    names = []
    rng = np.random.default_rng(0)
    for r in range(n_rows):
        baseline = 20 + r * row_mm
        xs = np.arange(int(x0 * PPMM), int((x0 + cols * seg_mm) * PPMM))
        wave = baseline * PPMM + np.where(rng.random(len(xs)) < 0.04, -rng.uniform(4, 20, len(xs)), 0)
        ink[np.clip(wave.round().astype(int), 0, h - 1), xs] = 1.0
        leads = rows[r] if r < len(rows) else [rhythm[r - len(rows)]]
        width = cols * seg_mm / len(leads)
        for c, lead in enumerate(leads):
            if lead not in drop:
                names.append(NameDetection(LEAD_INDEX[lead], x0 + c * width + 4.0, baseline + name_dy, 0.9))
    return ink, names


ROWS_3X4 = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
ROWS_6X2 = [["I", "V1"], ["II", "V2"], ["III", "V3"], ["aVR", "V4"], ["aVL", "V5"], ["aVF", "V6"]]


def summary(layout):
    return sorted((LEADS[s.lead], round(s.x0), round(s.x1), round(s.t0, 1)) for s in layout.segments)


def test_3x4_with_rhythm_strip():
    ink, names = make_page(ROWS_3X4, rhythm=["II"])
    layout = assemble_layout(ink, PPMM, names, 25.0)
    assert layout.template == "3x4+1R" and layout.reading == "names"
    assert len(layout.segments) == 13
    got = summary(layout)
    assert ("V1", 140, 202, 5.0) in got and ("I", 15, 78, 0.0) in got
    assert ("II", 15, 265, 0.0) in got  # the rhythm strip stays one 10 s segment
    assert len(layout.rows) == 4


def test_6x2_and_12x1():
    ink, names = make_page(ROWS_6X2, seg_mm=125.0, row_mm=22.0)
    layout = assemble_layout(ink, PPMM, names, 25.0)
    assert layout.template == "6x2" and len(layout.segments) == 12
    ink, names = make_page([[lead] for lead in LEADS], seg_mm=250.0, row_mm=14.0, name_dy=-4.0)
    layout = assemble_layout(ink, PPMM, names, 25.0)
    assert layout.template == "12x1" and len(layout.segments) == 12


def test_names_below_the_trace_vote_consistently():
    # names 9 mm *below* their trace with rows only 20 mm apart: each name is ambiguous alone
    ink, names = make_page(ROWS_6X2, seg_mm=125.0, row_mm=20.0, name_dy=9.0)
    layout = assemble_layout(ink, PPMM, names, 25.0)
    assert layout.template == "6x2"
    by_lead = {LEADS[s.lead]: s for s in layout.segments}
    assert abs(by_lead["I"].baseline - 20) < 1.5 and abs(by_lead["aVF"].baseline - 120) < 1.5


def test_missed_name_is_recovered_from_the_column_structure():
    ink, names = make_page(ROWS_3X4, rhythm=["II"], drop=("aVL", "V6"))
    layout = assemble_layout(ink, PPMM, names, 25.0)
    assert layout.template == "3x4+1R"
    got = {(LEADS[s.lead], s.source) for s in layout.segments}
    assert ("aVL", "template") in got and ("V6", "template") in got
    assert len(layout.segments) == 13


def test_cabrera_order_is_read_from_the_names():
    rows = [["aVL", "II", "V1", "V4"], ["I", "aVF", "V2", "V5"], ["aVR", "III", "V3", "V6"]]
    ink, names = make_page(rows)
    layout = assemble_layout(ink, PPMM, names, 25.0)
    assert layout.template == "3x4-cabrera"
    first_row = [LEADS[s.lead] for s in sorted(layout.segments, key=lambda s: (s.row, s.x0))[:4]]
    assert first_row == rows[0]


def test_speed_changes_time_not_geometry():
    ink, names = make_page(ROWS_3X4)
    a = assemble_layout(ink, PPMM, names, 25.0)
    b = assemble_layout(ink, PPMM, names, 50.0)
    assert [s.x0 for s in a.segments] == [s.x0 for s in b.segments]
    assert max(s.t0 for s in a.segments) == 2 * max(s.t0 for s in b.segments)


def test_unlabelled_page_falls_back_to_a_flagged_standard_layout():
    ink, _ = make_page(ROWS_3X4, rhythm=["II"])
    layout = assemble_layout(ink, PPMM, [], 25.0)
    assert layout.template == "3x4+1R" and layout.reading == "rows" and len(layout.segments) == 13
    assert any("assumed" in note for note in layout.notes)
    assert {s.source for s in layout.segments} == {"template", "assumed"}
    assert [a.template for a in layout.alternatives] == ["3x4-cabrera+1R"]
    empty, _, _ = layout_from_rows(np.zeros((100, 300), np.float32), PPMM, 25.0, LayoutConfig())
    assert not empty.segments


def test_a_few_names_are_enough_to_confirm_a_standard_layout():
    ink, names = make_page(ROWS_6X2, seg_mm=125.0, row_mm=22.0)
    few = [n for n in names if LEADS[n.lead] in ("II", "V4", "aVF")]
    layout = assemble_layout(ink, PPMM, few, 25.0)
    assert layout.template == "6x2" and len(layout.segments) == 12
    assert any("confirmed by 3" in note for note in layout.notes)
    got = summary(layout)
    assert ("V1", 140, 265, 5.0) in got and ("I", 15, 140, 0.0) in got


def test_few_names_distinguish_cabrera_from_standard():
    rows = [["aVL", "II", "V1", "V4"], ["I", "aVF", "V2", "V5"], ["aVR", "III", "V3", "V6"]]
    ink, names = make_page(rows)
    few = [n for n in names if LEADS[n.lead] in ("aVL", "III")]
    assert assemble_layout(ink, PPMM, few, 25.0).template == "3x4-cabrera"


def test_rhythm_strip_identity_comes_from_its_name():
    ink, names = make_page(ROWS_3X4, rhythm=["V1"])
    few = [n for n in names if LEADS[n.lead] in ("I", "V1")]
    layout = assemble_layout(ink, PPMM, few, 25.0)
    assert layout.template == "3x4+1R"
    strip = max(layout.segments, key=lambda s: s.x1 - s.x0)
    assert LEADS[strip.lead] == "V1" and round(strip.x1 - strip.x0) == 250


def test_non_standard_page_keeps_the_name_reading():
    rows = [["V1", "V2"], ["V3", "V4"]]
    ink, names = make_page(rows, seg_mm=125.0)
    layout = assemble_layout(ink, PPMM, names, 25.0)
    assert layout.template is None and len(layout.segments) == 4
    assert [LEADS[s.lead] for s in sorted(layout.segments, key=lambda s: (s.row, s.x0))] == ["V1", "V2", "V3", "V4"]


def test_six_rows_without_names_keep_both_readings_for_the_referee():
    ink, _ = make_page(ROWS_6X2, seg_mm=125.0, row_mm=22.0)
    layout = assemble_layout(ink, PPMM, [], 25.0)
    assert layout.template == "6x2"
    assert {a.template for a in layout.alternatives} == {
        "3x4+3R", "6x2-cabrera", "3x4-cabrera+3R", "6x1-limb", "6x1-chest"}


def test_regular_rows_drop_borders_and_text_lines():
    from ecgdl.layout.assemble import regular_rows

    real = [(40.0 + 28 * r, 15.0, 265.0) for r in range(6)]
    spurious = [(6.0, 0.0, 280.0)] + real + [(40.0 + 28 * 5 + 17, 20.0, 250.0)]
    assert regular_rows(spurious) == real
    assert regular_rows(real) == real
    assert regular_rows(real[:2]) == real[:2]


def test_two_block_6x2_page_gets_per_column_rows():
    from ecgdl.layout.assemble import split_column_hypotheses

    h, w = int(170 * PPMM), int(290 * PPMM)
    ink = np.zeros((h, w), np.float32)
    left = [20 + 10 * r for r in range(6)]  # limb block: tight rows
    right = [30 + 22 * r for r in range(6)]  # chest block: wide rows, different baselines
    for b in left:
        ink[int(b * PPMM), int(15 * PPMM) : int(140 * PPMM)] = 1.0
    for b in right:
        ink[int(b * PPMM), int(140 * PPMM) : int(265 * PPMM)] = 1.0
    hyps = split_column_hypotheses(ink, PPMM, 25.0, LayoutConfig())
    assert [x.template for x in hyps] == ["6x2", "6x2-cabrera"]
    std = hyps[0]
    by_lead = {LEADS[s.lead]: s for s in std.segments}
    assert abs(by_lead["I"].baseline - 20) < 1 and abs(by_lead["aVF"].baseline - 70) < 1
    assert abs(by_lead["V1"].baseline - 30) < 1 and abs(by_lead["V6"].baseline - 140) < 1
    assert by_lead["I"].t0 == 0.0 and by_lead["V1"].t0 == 5.0
    assert len({s.row for s in std.segments}) == 12
    # an ordinary 6x2 page (shared rows) is left to the standard path
    ink2, _ = make_page(ROWS_6X2, seg_mm=125.0, row_mm=22.0)
    assert split_column_hypotheses(ink2, PPMM, 25.0, LayoutConfig()) == []
