language_translator

```
This project is an automated translation pipeline that converts a medical 
QA dataset (MedQA) from English into German and French. It supports two 
translation backends — the free Google Translate via deep-translator, or 
the higher-quality DeepL API for users with an API key.

The pipeline includes entity protection to shield numbers, dates, emails, 
URLs, and proper names from being mistranslated, exponential back-off retry 
logic for handling API failures, and a validation layer that checks each 
translation for truncation, leftover English words, German noun 
capitalisation, and back-translation confidence scoring.

Each translated dataset is saved as a separate Excel file, and any 
flagged issues are compiled into a translation report for manual review.

Built with Python, pandas, deep-translator, and optionally DeepL.
