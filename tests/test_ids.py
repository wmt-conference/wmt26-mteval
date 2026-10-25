import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from wmt26_mteval.context import Context, LangCodes
from wmt26_mteval.ids import (
    SYSTEM_TYPES,
    ItemId,
    challenge_set,
    classify_mt_system,
    domain,
    is_reference_candidate,
    map_ref_type,
    parse_item_id,
)

PRIORITY_LPS = [
    "ces_Latn-deu_Latn",
    "ces_Latn-vie_Latn",
    "eng_Latn-arz_Arab",
    "eng_Latn-ekk_Latn",
    "eng_Latn-ind_Latn",
    "eng_Latn-isl_Latn",
    "eng_Latn-sme_Latn",
    "zho_Hans-jpn_Jpan",
]


def test_parse_official():
    i = parse_item_id(
        "official_###_ces_Latn_###_deu_Latn_###_speech_###_id_rtWATtjAFUM_29.58-58.15_###_0"
    )
    assert (i.collection, i.field4, i.doc_id, i.para_idx) == (
        "official",
        "speech",
        "id_rtWATtjAFUM_29.58-58.15",
        0,
    )


def test_parse_rejects_bad():
    with pytest.raises(ValueError):
        parse_item_id("a_###_b")


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "official_###_ces_Latn_###_deu_Latn_###_news_###_d",  # 5 parts
        "official_###_ces_Latn_###_deu_Latn_###_news_###_d_###_0_###_1",  # 7 parts
        "official_###_ces_Latn_###_deu_Latn_###_news_###_d_###_x",  # non-integer paragraph
        "official_###_ces_Latn_###_deu_Latn_###_news_###_d_###_",  # empty paragraph
        "challenge_###_en_###_de_###_GAMBITP_###_8111|24041_###_abc",  # hashes: FECE/QEbreak only
        "challenge_###_en_###_de_###_FECE_###_d_###_",  # empty segment hash
    ],
)
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_item_id(bad)


def test_parse_fields_and_types():
    i = parse_item_id("challenge_###_en_###_de_DE_###_LIGATUR_###_adjg-hi-01_###_00064")
    assert i == ItemId("challenge", "en", "de_DE", "LIGATUR", "adjg-hi-01", 64)
    assert isinstance(i.para_idx, int)


@pytest.mark.parametrize(
    "item_id",
    [
        "challenge_###_cs_###_uk_###_QEbreak_###_QEbreak_01301a2135ae198c_###_74d5933a8605",
        # 35 real FECE/QEbreak hashes are all digits; they must not become (int32-overflowing) ints
        "challenge_###_en_###_ar_###_QEbreak_###_QEbreak_cd739e3e91fc15e4_###_930582188412",
        "challenge_###_en_###_de_DE_###_FECE_###_test-en-literary_x_chunk_1|2|794|Dubformer_###_28e183a775",
        "challenge_###_en_###_ja_JP_###_FECE_###_test-en-news_economist.14223|20|GPT-4_###_1744613681",
    ],
)
def test_parse_segment_hash_has_no_paragraph_index(item_id):
    i = parse_item_id(item_id)
    assert i.field4 in {"FECE", "QEbreak"} and i.para_idx is None


@pytest.mark.parametrize(
    "doc,field4,expected",
    [
        ("RefQuality-PE_id_x", "speech", "RefQuality-PE"),
        ("RefQuality-PD_RefQuality-PE_edu01691", "edu", "RefQuality-PD"),
        ("8111|24041", "GAMBITP", "GAMBITP"),
    ],
)
def test_challenge_set(doc, field4, expected):
    assert challenge_set(ItemId("challenge", "en", "de", field4, doc, 0)) == expected


@pytest.mark.parametrize("field4", ["FECE", "QEbreak", "AGGREE", "LIGATUR", "GAMBITP"])
def test_challenge_set_named_by_field4(field4):
    assert challenge_set(ItemId("challenge", "en", "de", field4, "any_doc", 0)) == field4


def test_challenge_set_official_is_none():
    official = ItemId("official", "ces_Latn", "deu_Latn", "news", "RefQuality-PE_x", 0)
    assert challenge_set(official) is None


def test_challenge_set_unrecognised_challenge_item_fails_loudly():
    with pytest.raises(ValueError):
        challenge_set(ItemId("challenge", "en", "de", "news", "plain_doc", 0))
    with pytest.raises(ValueError):
        challenge_set(ItemId("shared", "en", "de", "news", "d", 0))


def test_domain_only_for_official_and_refquality():
    assert domain(ItemId("challenge", "en", "de", "FECE", "test-en-news_1", 0)) is None
    assert domain(ItemId("challenge", "ces_Latn", "deu_Latn", "edu", "RefQuality-PD_x", 0)) == "edu"


def test_domain_official_and_other_challenge_sets():
    assert domain(ItemId("official", "ces_Latn", "deu_Latn", "software", "d", 0)) == "software"
    assert domain(ItemId("challenge", "en", "ar", "GAMBITP", "0110|28892", 8610)) is None
    assert domain(ItemId("challenge", "en", "ko", "QEbreak", "QEbreak_00c9", None)) is None


@pytest.mark.parametrize(
    "ref,out",
    [
        (None, ("none", None)),
        ({"text": None, "type": None}, ("none", None)),
        ({"text": "x", "type": "undefined"}, ("unknown", "undefined")),
        ({"text": None, "type": "undefined"}, ("none", "undefined")),
        ({"text": "x", "type": None}, ("unknown", None)),
        ({"text": "x", "type": "human"}, ("native", "human")),
        ({"text": "x", "type": "postedit"}, ("postedit", "postedit")),
        ({"text": "x", "type": "pseudo"}, ("pseudo", "pseudo")),
    ],
)
def test_map_ref_type(ref, out):
    assert map_ref_type(ref) == out


def test_map_ref_type_rejects_unknown_type():
    with pytest.raises(ValueError):
        map_ref_type({"text": "x", "type": "gold"})


@pytest.mark.parametrize(
    "name,t",
    [
        ("Human (from scratch)", "human_scratch"),
        ("Human (postediting)", "human_postedit"),
        ("PseudoRef", "pseudo_ref"),
        ("FECE_hyp2", "challenge_hyp"),
        ("Cohere CAT+", "mt"),
    ],
)
def test_classify(name, t):
    assert classify_mt_system(name) == t


@pytest.mark.parametrize(
    "name",
    ["AGGREE_hyp1", "FECE_hyp1", "GAMBITP_hyp1", "GAMBITP_hyp2", "LIGATUR_hyp2", "QEbreak_hyp1"],
)
def test_classify_every_challenge_hypothesis_name(name):
    assert classify_mt_system(name) == "challenge_hyp"


@pytest.mark.parametrize("name", ["Gemma 4 - 4B", "Yandex-8B", "VoxNexus_V1", "Humanoid", "HW-TSC"])
def test_classify_other_names_are_mt(name):
    assert classify_mt_system(name) == "mt"


def test_system_types_are_exactly_what_classify_returns():
    assert SYSTEM_TYPES == ("mt", "human_scratch", "human_postedit", "pseudo_ref", "challenge_hyp")
    names = ["Cohere CAT+", "Human (from scratch)", "Human (postediting)", "PseudoRef", "FECE_hyp1"]
    assert [classify_mt_system(name) for name in names] == list(SYSTEM_TYPES)


def test_reference_candidate_ignores_outer_whitespace():  # Review Focus 2
    assert is_reference_candidate("Hallo Welt\n ", "Hallo Welt")
    assert not is_reference_candidate("Hallo  Welt", "Hallo Welt")
    assert not is_reference_candidate("x", None)


def test_reference_candidate_exact_and_leading_whitespace():
    assert is_reference_candidate("Hallo Welt", "Hallo Welt")
    assert is_reference_candidate("  Hallo Welt", "Hallo Welt\t")
    assert not is_reference_candidate("hallo welt", "Hallo Welt")


def test_lp_short_matches_paper(repo_root):
    ctx = Context.from_repo(repo_root)

    def lp(iid):
        return ctx.describe(iid)["lp"]

    assert lp("official_###_eng_Latn_###_zho_Hans_###_news_###_d_###_0") == "eng-zho_CN"
    assert lp("official_###_eng_Latn_###_zho_Hant_TW_###_news_###_d_###_0") == "eng-zho_TW"
    assert lp("official_###_zho_Hans_###_jpn_Jpan_###_news_###_d_###_0") == "zho-jpn"
    assert lp("challenge_###_en_###_de_DE_###_FECE_###_d_###_0") == "eng-deu"


def test_priority_is_official_only(repo_root):
    ctx = Context.from_repo(repo_root)
    assert ctx.describe("official_###_ces_Latn_###_deu_Latn_###_news_###_d_###_0")["is_priority"]
    challenge = "challenge_###_ces_Latn_###_deu_Latn_###_news_###_RefQuality-PE_d_###_0"
    assert not ctx.describe(challenge)["is_priority"]


def test_ctx_fixture_is_a_context(ctx):
    assert isinstance(ctx, Context)
    assert ctx.priority == frozenset(PRIORITY_LPS)


def test_priority_covers_the_eight_official_directions(ctx):
    for lp in PRIORITY_LPS:
        src, tgt = lp.split("-")
        assert ctx.describe(f"official_###_{src}_###_{tgt}_###_news_###_d_###_0")["is_priority"], lp
    other = "official_###_eng_Latn_###_deu_Latn_###_news_###_d_###_0"
    assert not ctx.describe(other)["is_priority"]


def test_describe_exact_keys_and_values(ctx):
    d = ctx.describe("challenge_###_en_###_de_DE_###_FECE_###_test-en-news_1|2|3|X_###_28e183a775")
    assert list(d) == [
        "collection",
        "challenge_set",
        "domain",
        "src_lang",
        "tgt_lang",
        "src_lang_orig",
        "tgt_lang_orig",
        "lp",
        "is_priority",
        "doc_id",
        "para_idx",
    ]
    assert d == {
        "collection": "challenge",
        "challenge_set": "FECE",
        "domain": None,
        "src_lang": "eng_Latn",
        "tgt_lang": "deu_Latn",
        "src_lang_orig": "en",
        "tgt_lang_orig": "de_DE",
        "lp": "eng-deu",
        "is_priority": False,
        "doc_id": "test-en-news_1|2|3|X",
        "para_idx": None,
    }


def test_describe_official(ctx):
    d = ctx.describe("official_###_ces_Latn_###_ukr_Cyrl_###_edu_###_edu01691_###_0")
    assert (d["collection"], d["challenge_set"], d["domain"]) == ("official", None, "edu")
    assert (d["src_lang"], d["tgt_lang"], d["lp"]) == ("ces_Latn", "ukr_Cyrl", "ces-ukr")
    assert d["para_idx"] == 0
    assert d["src_lang_orig"] == "ces_Latn" and d["tgt_lang_orig"] == "ukr_Cyrl"


def test_describe_unknown_language_code_fails_loudly(ctx):
    with pytest.raises(KeyError):
        ctx.describe("official_###_xx_Latn_###_deu_Latn_###_news_###_d_###_0")


def test_lang_codes_normalise_and_short(repo_root):
    lang = LangCodes.load(repo_root / "data" / "metadata" / "lang_codes.tsv")
    mapping = {
        "en": "eng_Latn",
        "de": "deu_Latn",
        "de_DE": "deu_Latn",
        "cs": "ces_Latn",
        "uk": "ukr_Cyrl",
        "ar": "arb_Arab",
        "el": "ell_Grek",
        "is": "isl_Latn",
        "ru": "rus_Cyrl",
        "hi": "hin_Deva",
        "ja": "jpn_Jpan",
        "ja_JP": "jpn_Jpan",
        "ko": "kor_Hang",
        "zh": "zho_Hans",
        "zh_CN": "zho_Hans",
        "arz_Arab": "arz_Arab",
        "zho_Hant_TW": "zho_Hant_TW",
    }
    for orig, norm in mapping.items():
        assert lang.normalise(orig) == norm, orig
    with pytest.raises(KeyError):
        lang.normalise("fr")
    assert [lang.short("zho_Hans", "src"), lang.short("zho_Hans", "tgt")] == ["zho", "zho_CN"]
    assert [lang.short("zho_Hant_TW", r) for r in ("src", "tgt")] == ["zho_TW", "zho_TW"]
    assert lang.short("ces_Latn", "src") == lang.short("ces_Latn", "tgt") == "ces"


def test_lang_codes_load_skips_header_and_blank_lines(tmp_path):
    p = tmp_path / "codes.tsv"
    p.write_text("orig\tnormalised\nen\teng_Latn\n\nde\tdeu_Latn\n", encoding="utf-8")
    lang = LangCodes.load(p)
    assert lang.normalise("en") == "eng_Latn" and lang.normalise("de") == "deu_Latn"
    with pytest.raises(KeyError):
        lang.normalise("orig")


@pytest.fixture(scope="module")
def real_item_ids():
    grand = os.environ.get("GRAND")
    if not grand or not Path(grand).is_file():
        pytest.skip("GRAND is not set to the real wmt26-grand.parquet")
    return pq.read_table(grand, columns=["item_id"]).column(0).to_pylist()


@pytest.mark.fulldata
def test_every_real_language_code_is_mapped(repo_root, real_item_ids):
    lang = LangCodes.load(repo_root / "data" / "metadata" / "lang_codes.tsv")
    codes = set()
    for item_id in real_item_ids:
        i = parse_item_id(item_id)
        codes.update((i.src_lang_orig, i.tgt_lang_orig))
    assert len(codes) == 35
    for code in sorted(codes):
        lang.normalise(code)


@pytest.mark.fulldata
def test_every_real_item_id_is_described(ctx, real_item_ids):
    assert len(real_item_ids) == 26885
    described = [ctx.describe(item_id) for item_id in real_item_ids]
    for d in described:
        assert (d["challenge_set"] is None) == (d["collection"] == "official")
        assert d["para_idx"] is None or 0 <= d["para_idx"] < 2**31
    assert {d["challenge_set"] for d in described} == {
        None,
        "AGGREE",
        "FECE",
        "GAMBITP",
        "LIGATUR",
        "QEbreak",
        "RefQuality-PD",
        "RefQuality-PE",
    }
    assert sum(d["para_idx"] is None for d in described) == 5886  # FECE 3000 + QEbreak 2886
