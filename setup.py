# ============================================================
# SECTION 2 — TRANSLATOR SETUP
# ============================================================

from config import DEEPL_API_KEY, log


def get_translator():
    """
    Returns a (translator_type, translator) tuple.
    Prefers DeepL if DEEPL_API_KEY is set, otherwise falls back to Google Translate.
    """
    if DEEPL_API_KEY:
        try:
            import deepl
            log.info("Using DeepL Translator")
            return "deepl", deepl.Translator(DEEPL_API_KEY)
        except ImportError:
            log.warning("'deepl' package not installed. Run: pip install deepl")
            log.info("Falling back to Google Translate...")

    try:
        from deep_translator import GoogleTranslator
        log.info("Using Google Translator (free, no API key required)")
        log.info("For better quality, set DEEPL_API_KEY in your environment")
        return "google", GoogleTranslator
    except ImportError:
        raise ImportError(
            "No translation package found.\n"
            "Install one of:\n"
            "  pip install deepl\n"
            "  pip install deep_translator"
        )

# Initialise once at import time — all modules import from here
translator_type, translator = get_translator()