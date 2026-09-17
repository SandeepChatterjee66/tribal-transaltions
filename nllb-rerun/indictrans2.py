#!/usr/bin/env python3
"""
indictrans2.py — adapter for the AI4Bharat IndicTrans2 models.

IndicTrans2 does not follow the plain HuggingFace translation convention: it
expects text to be normalised and tagged with explicit `<src> <tgt>` language
tokens by its own IndicProcessor, and the output has to be post-processed to
undo the entity placeholders it inserts. Everything specific to that lives
here so `infer.py` and `finetune.py` stay model-agnostic.

Why it is worth the extra file: IndicTrans2 is the strongest and most expected
Indic baseline, it is already cited in the paper's related work, and it is the
only model in the registry that knows Santali in Ol Chiki (`sat_Olck`) — the
script TribalCorp actually uses. NLLB-200 only has `sat_Beng`.

Install:
    pip install IndicTransToolkit

If it is missing, this module raises with the install command rather than
failing somewhere confusing inside generate().
"""

from __future__ import annotations

from common import LOG

# 22 scheduled languages + English. Kept here for error messages; the real
# check is always against the loaded tokenizer.
INDICTRANS2_CODES = {
    "asm_Beng", "ben_Beng", "brx_Deva", "doi_Deva", "eng_Latn", "gom_Deva",
    "guj_Gujr", "hin_Deva", "kan_Knda", "kas_Arab", "kas_Deva", "mai_Deva",
    "mal_Mlym", "mar_Deva", "mni_Beng", "mni_Mtei", "npi_Deva", "ory_Orya",
    "pan_Guru", "san_Deva", "sat_Olck", "snd_Arab", "snd_Deva", "tam_Taml",
    "tel_Telu", "urd_Arab",
}


def _require_toolkit():
    try:
        from IndicTransToolkit.processor import IndicProcessor  # noqa: F401
        return IndicProcessor
    except ImportError:
        pass
    try:
        from IndicTransToolkit import IndicProcessor  # older layout
        return IndicProcessor
    except ImportError:
        raise ImportError(
            "IndicTrans2 needs the AI4Bharat toolkit:\n"
            "    pip install IndicTransToolkit\n"
            "Only the indictrans2-* models require it; NLLB / mBART / M2M100 "
            "run without it. Drop indictrans2 from MODELS to skip."
        ) from None


class IndicTrans2Adapter:
    """Wraps IndicProcessor so the rest of the codebase can ignore it.

    Usage mirrors what infer/finetune need:
        ad = IndicTrans2Adapter()
        srcs = ad.preprocess(texts, "hin_Deva", "eng_Latn")
        ...  generate  ...
        hyps = ad.postprocess(decoded, "eng_Latn")
    """

    def __init__(self):
        IndicProcessor = _require_toolkit()
        # inference=True applies the entity-placeholder logic IndicTrans2
        # was trained with; without it quality drops noticeably.
        self.ip = IndicProcessor(inference=True)
        LOG.info("IndicTrans2 processor ready")

    def preprocess(self, texts: list[str], src_lang: str,
                   tgt_lang: str) -> list[str]:
        """Normalise + prepend the `<src> <tgt>` tags IndicTrans2 expects."""
        return self.ip.preprocess_batch(
            [str(t) for t in texts], src_lang=src_lang, tgt_lang=tgt_lang)

    def postprocess(self, decoded: list[str], tgt_lang: str) -> list[str]:
        """Restore entities and detokenise."""
        out = self.ip.postprocess_batch(list(decoded), lang=tgt_lang)
        return [str(x).strip() for x in out]


def is_indictrans2(spec) -> bool:
    return getattr(spec, "family", None) == "indictrans2"


def check_codes(spec, languages: list[str]) -> None:
    """Warn early about codes IndicTrans2 is known not to support.

    Advisory only — the authoritative check is the tokenizer validation in
    common.validate_lang_token, which runs against the real vocabulary.
    """
    unknown = {}
    for lang in languages:
        code = spec.source_tokens.get(lang)
        if code and code not in INDICTRANS2_CODES:
            unknown[lang] = code
    if unknown:
        LOG.warning(
            "IndicTrans2 does not list these source codes: %s. It supports "
            "only the 22 scheduled languages. For Bhili/Gondi/Mundari a "
            "Devanagari proxy such as hin_Deva is the intended fallback.",
            unknown)
    if spec.source_tokens.get("Santali") == "sat_Olck":
        LOG.info("IndicTrans2 supports sat_Olck natively — the only model here "
                 "that knows Santali in Ol Chiki (NLLB has only sat_Beng)")
