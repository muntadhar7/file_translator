from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import polib
import uuid
import requests
import time
import json
import csv
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple
import asyncio
import aiohttp
from fastapi.responses import Response
import re
import html
from fastapi.responses import HTMLResponse


from collections import OrderedDict

app = FastAPI(
    title="Universal File Translator",
    description="Translate multiple file formats using AI translation APIs",
    version="2.0.0"
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Supported file types
SUPPORTED_EXTENSIONS = {
    '.po': 'Gettext PO Files',
    '.json': 'JSON Files',
    '.csv': 'CSV Files',
    '.xlf': 'XLIFF Files',
    '.xliff': 'XLIFF Files',
    '.xml': 'XML Files',
    '.txt': 'Text Files',
    '.properties': 'Java Properties'
}

# Optimized translation services
TRANSLATION_SERVICES = {
    "google_translate": {
        "url": "https://translate.googleapis.com/translate_a/single",
        "method": "GET",
        "batch_support": True,
        "max_texts_per_request": 100,
        "max_chars_per_request": 5000,
    },
    "mymemory_batch": {
        "url": "https://api.mymemory.translated.net/get",
        "method": "GET",
        "batch_support": False,
        "max_chars": 500
    },
    "libretranslate_batch": {
        "url": "https://libretranslate.com/translate",
        "method": "POST",
        "batch_support": True,
        "max_texts_per_request": 50,
        "max_chars_per_request": 5000
    }
}

# Supported languages
SUPPORTED_LANGUAGES = {
    "ar": "Arabic", "es": "Spanish", "fr": "French", "de": "German",
    "zh": "Chinese", "ja": "Japanese", "ru": "Russian", "it": "Italian",
    "pt": "Portuguese", "hi": "Hindi", "ko": "Korean", "tr": "Turkish",
    "nl": "Dutch", "pl": "Polish", "sv": "Swedish", "da": "Danish"
}

# --- Helper: mask placeholders/format tokens and HTML tags ---
# NOTE: removed aggressive parenthesis pattern to avoid masking code expressions.
PLACEHOLDER_PATTERNS = [
    # Python percent formatting: %(name)s, %s, %d, %0.2f
    r'%\([^\)]+\)s', r'%[-+]?\d*\.\d+[sdif]', r'%[sdif]',
    # Curly braces and moustache-like templates: {name}, {{escaped}}
    r'\{\{.*?\}\}', r'\{[^}]+\}',
    # Jinja style blocks - keep as tokens
    r'\{%-?.*?-%\}',
    # XML/HTML tags (protect tags and attributes)
    r'</?[\w:\-]+(?:\s+[^>]+)?>',
    # URLs and emails (protect them too)
    r'https?://[^\s]+', r'\b[\w\.-]+@[\w\.-]+\.\w+\b'
]

# compile combined regex
_COMPILED_MASK_RE = re.compile('|'.join('(?:%s)' % p for p in PLACEHOLDER_PATTERNS), flags=re.DOTALL)


def mask_text(text: str):
    """
    Replace matched tokens with numeric placeholders __PH_{n}__ and return:
      masked_text, placeholders_list
    placeholders_list is list of original tokens in order for this single text.
    Note: Masking is intended to be used on unique texts (deduplicated set).
    """
    if not text:
        return text, []

    placeholders = []

    def _repl(m):
        idx = len(placeholders)
        token = m.group(0)
        placeholders.append(token)
        return f"__PH_{idx}__"

    masked = _COMPILED_MASK_RE.sub(_repl, text)
    # Escape any leftover angle brackets so the translator doesn't interpret HTML
    # but preserve placeholder tokens
    parts = re.split(r'(__PH_\d+__)', masked)
    escaped_parts = []
    for p in parts:
        if p.startswith("__PH_") and p.endswith("__"):
            escaped_parts.append(p)
        else:
            escaped_parts.append(html.escape(p))
    result = ''.join(escaped_parts)
    return result, placeholders


def unmask_text(masked_text: str, placeholders: list):
    """Restore placeholders back into masked_text and unescape HTML entities."""
    if not masked_text:
        return masked_text

    def _repl(m):
        idx = int(m.group(1))
        if 0 <= idx < len(placeholders):
            return placeholders[idx]
        # if out-of-range, return the token unchanged so it is visible for debugging
        return m.group(0)

    restored = re.sub(r'__PH_(\d+)__', _repl, masked_text)
    # Unescape HTML entities translated by the service
    restored = html.unescape(restored)
    return restored

# --- New helper: deduplicate preserving indices ---
def dedupe_preserve_indices(texts: List[str]):
    """
    Return (unique_texts, occurrences) where:
      unique_texts: list of unique texts in first-seen order
      occurrences: dict mapping unique_index -> list of original indices where it appeared
    """
    unique_index = {}
    unique_texts = []
    occurrences = {}

    for i, t in enumerate(texts):
        if t in unique_index:
            idx = unique_index[t]
            occurrences[idx].append(i)
        else:
            idx = len(unique_texts)
            unique_index[t] = idx
            unique_texts.append(t)
            occurrences[idx] = [i]

    return unique_texts, occurrences
# -------------------------------------------------


def remove_file(path: str):
    """Remove temporary files"""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"Error removing file {path}: {e}")


def clean_translation_text(text: str) -> str:
    """Clean translation text by removing trailing newlines and extra spaces"""
    if not text:
        return text
    return text.rstrip('\n\r ')


def create_optimized_batches(texts: List[str], max_chars: int = 4500, max_texts: int = 100) -> List[List[str]]:
    """Create optimized batches considering both character count and text count"""
    batches = []
    current_batch = []
    current_chars = 0

    for text in texts:
        text_chars = len(text)

        # If single text is too large, put it in its own batch
        if text_chars > max_chars:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            batches.append([text])
            continue

        # Check if we can add to current batch
        if (current_chars + text_chars <= max_chars and
                len(current_batch) < max_texts):
            current_batch.append(text)
            current_chars += text_chars
        else:
            # Start new batch
            if current_batch:
                batches.append(current_batch)
            current_batch = [text]
            current_chars = text_chars

    # Add the last batch
    if current_batch:
        batches.append(current_batch)

    print(f"Created {len(batches)} batches for {len(texts)} texts")
    return batches


def create_optimized_batches_indices(texts: List[str], max_chars: int = 4500, max_texts: int = 100) -> List[List[int]]:
    """Create optimized batches of indices (preserves original order and handles duplicates)"""
    batches: List[List[int]] = []
    current_batch: List[int] = []
    current_chars = 0

    for idx, text in enumerate(texts):
        text_chars = len(text)

        # If single text is too large, put it in its own batch
        if text_chars > max_chars:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            batches.append([idx])
            continue

        # Check if we can add to current batch
        if (current_chars + text_chars <= max_chars and
                len(current_batch) < max_texts):
            current_batch.append(idx)
            current_chars += text_chars
        else:
            # Start new batch
            if current_batch:
                batches.append(current_batch)
            current_batch = [idx]
            current_chars = text_chars

    # Add the last batch
    if current_batch:
        batches.append(current_batch)

    print(f"Created {len(batches)} batches for {len(texts)} texts (index-based)")
    return batches


def parse_google_translate_response(response_data: List, original_texts: List[str]) -> List[str]:
    """Parse Google Translate API response.

    Returns a list of translated strings if we can extract any translations.
    It may return a shorter list (partial) — caller will verify usefulness.
    """
    translations: List[str] = []

    try:
        if not response_data:
            return []

        # Typical structure: response_data[0] is a list of [translatedText, ...] items.
        if isinstance(response_data, list) and len(response_data) > 0:
            main_data = response_data[0]

            # If main_data is a list of lists -> per-segment translations
            if isinstance(main_data, list):
                for item in main_data:
                    if item and isinstance(item, list) and len(item) > 0:
                        translated_text = item[0]
                        if translated_text and isinstance(translated_text, str):
                            translations.append(clean_translation_text(translated_text))

                if translations:
                    return translations

        # Fallback: traverse the structure looking for the first string blob,
        # attempt to split by newline into pieces.
        def find_first_string(obj):
            if isinstance(obj, str):
                return obj
            if isinstance(obj, list):
                for el in obj:
                    res = find_first_string(el)
                    if res:
                        return res
            if isinstance(obj, dict):
                for v in obj.values():
                    res = find_first_string(v)
                    if res:
                        return res
            return None

        candidate = find_first_string(response_data)
        if candidate:
            candidate = clean_translation_text(candidate)
            parts = candidate.split('\n')
            if parts:
                return [clean_translation_text(p) for p in parts]

    except Exception as e:
        print(f"Error parsing Google Translate response: {e}")

    # Return empty list to indicate "no translations extracted"
    return []


async def translate_with_google_batch(texts: List[str], target_lang: str, session: aiohttp.ClientSession) -> List[str]:
    """Translate batch using Google Translate API.

    Strategy:
     - Try repeated 'q' params request first (preferred for per-segment responses).
     - If the returned translations are not useful (very few differences), retry with a single combined q.
     - Only one retry to avoid large slowdowns.
    """
    if not texts:
        return []

    url = TRANSLATION_SERVICES["google_translate"]["url"]

    def extract_and_score_sync(response_json):
        # parse_google_translate_response is synchronous and light-weight; keep this sync.
        parsed = parse_google_translate_response(response_json, texts)
        if not parsed:
            return parsed, 0
        length = min(len(parsed), len(texts))
        diffs = sum(1 for i in range(length) if clean_translation_text(parsed[i]) != texts[i])
        return parsed, diffs

    try:
        # Attempt 1: repeated 'q' parameters
        params_list = [
            ("client", "gtx"),
            ("dt", "t"),
            ("sl", "auto"),
            ("tl", target_lang),
        ]
        for t in texts:
            params_list.append(("q", t))

        async with session.get(url, params=params_list, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 200:
                data = await resp.json()
                parsed, diffs = extract_and_score_sync(data)
                # If there is at least one real difference (or a decent proportion), accept it.
                threshold = 1 if len(texts) <= 10 else max(1, int(0.10 * len(texts)))
                if diffs >= threshold and parsed:
                    if len(parsed) < len(texts):
                        result = []
                        for i in range(len(texts)):
                            if i < len(parsed):
                                result.append(clean_translation_text(parsed[i]))
                            else:
                                result.append(texts[i])
                        return result
                    return [clean_translation_text(x) for x in parsed[:len(texts)]]

        # Attempt 2: single combined q (join by newline)
        combined_text = "\n".join(texts)
        params = {
            "client": "gtx",
            "dt": "t",
            "sl": "auto",
            "tl": target_lang,
            "q": combined_text
        }

        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp2:
            if resp2.status == 200:
                data2 = await resp2.json()
                parsed2 = parse_google_translate_response(data2, texts)
                if parsed2:
                    if len(parsed2) == len(texts):
                        return [clean_translation_text(x) for x in parsed2]
                    if len(parsed2) == 1 and '\n' in parsed2[0]:
                        parts = [clean_translation_text(p) for p in parsed2[0].split('\n')]
                        if len(parts) == len(texts):
                            return parts
                    result = []
                    for i in range(len(texts)):
                        if i < len(parsed2):
                            result.append(clean_translation_text(parsed2[i]))
                        else:
                            result.append(texts[i])
                    return result

    except Exception as e:
        print(f"Google batch translation failed: {e}")

    # Final fallback: return originals (caller may try other services)
    return texts

async def translate_with_libretranslate_batch(texts: List[str], target_lang: str, session: aiohttp.ClientSession) -> \
List[str]:
    """Translate batch using LibreTranslate"""
    try:
        payload = {
            "q": texts,
            "source": "en",
            "target": target_lang,
            "format": "text"
        }

        async with session.post(
                TRANSLATION_SERVICES["libretranslate_batch"]["url"],
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                data = await response.json()
                translated_texts = data.get("translatedText", texts)

                if len(translated_texts) == len(texts):
                    return [clean_translation_text(text) for text in translated_texts]
                else:
                    return texts
            else:
                return texts
    except Exception as e:
        print(f"LibreTranslate batch translation failed: {e}")
        return texts


def translate_with_mymemory_single(text: str, target_lang: str) -> str:
    """Translate single text using MyMemory (synchronous fallback)"""
    try:
        if len(text) > TRANSLATION_SERVICES["mymemory_batch"]["max_chars"]:
            return text

        params = {
            "q": text,
            "langpair": f"en|{target_lang}",
            "de": "po-translator@example.com"
        }

        response = requests.get(
            TRANSLATION_SERVICES["mymemory_batch"]["url"],
            params=params,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("responseStatus") == 200:
                translated = data["responseData"]["translatedText"]
                if translated and translated != text:
                    return clean_translation_text(translated)
        return text
    except Exception as e:
        print(f"MyMemory translation failed: {e}")
        return text

async def translate_with_mymemory_single_async(text: str, target_lang: str, session: aiohttp.ClientSession) -> str:
    """Async version of MyMemory single translation using aiohttp to avoid blocking."""
    try:
        if len(text) > TRANSLATION_SERVICES["mymemory_batch"]["max_chars"]:
            return text

        params = {
            "q": text,
            "langpair": f"en|{target_lang}",
            "de": "po-translator@example.com"
        }

        async with session.get(
            TRANSLATION_SERVICES["mymemory_batch"]["url"],
            params=params,
            timeout=aiohttp.ClientTimeout(total=10)
        ) as response:
            if response.status == 200:
                data = await response.json()
                if data.get("responseStatus") == 200:
                    translated = data["responseData"].get("translatedText")
                    if translated and translated != text:
                        return clean_translation_text(translated)
        return text
    except Exception as e:
        print(f"MyMemory async translation failed: {e}")
        return text


async def process_batch_async(batch: List[str], target_lang: str, session: aiohttp.ClientSession) -> List[str]:
    """Process a single batch asynchronously"""
    if not batch:
        return []

    # Try LibreTranslate first
    result = await translate_with_libretranslate_batch(batch, target_lang, session)
    if result != batch:
        return result

    # Fallback to Google Translate
    result = await translate_with_google_batch(batch, target_lang, session)
    if result != batch:
        return result

    # Final fallback: individual translations (async)
    individual_results = []
    for text in batch:
        translated = await translate_with_mymemory_single_async(text, target_lang, session)
        individual_results.append(translated)
    return individual_results

async def retry_untranslated_async(failed_indices: List[int], texts: List[str], target_lang: str, session: aiohttp.ClientSession, concurrency: int = 6) -> Dict[int, str]:
    """Retry translations for only failed indices using alternative services/shapes.
    Returns a dict mapping index -> new_translation for items that improved.
    """
    results: Dict[int, str] = {}
    sem = asyncio.Semaphore(concurrency)

    async def retry_one(idx: int):
        async with sem:
            text = texts[idx]
            # Try Google with combined-q (if per-batch used repeated q earlier, this is alternative)
            try:
                google_try = await translate_with_google_batch([text], target_lang, session)
                if google_try and google_try[0] and clean_translation_text(google_try[0]) != text:
                    return idx, clean_translation_text(google_try[0])
            except Exception:
                pass

            # Try LibreTranslate per-item
            try:
                libre_try = await translate_with_libretranslate_batch([text], target_lang, session)
                if libre_try and libre_try[0] and clean_translation_text(libre_try[0]) != text:
                    return idx, clean_translation_text(libre_try[0])
            except Exception:
                pass

            # Final try: MyMemory async
            try:
                mm_try = await translate_with_mymemory_single_async(text, target_lang, session)
                if mm_try and clean_translation_text(mm_try) != text:
                    return idx, clean_translation_text(mm_try)
            except Exception:
                pass

            # Nothing improved
            return idx, text

    tasks = [retry_one(i) for i in failed_indices]
    for coro in asyncio.as_completed(tasks):
        try:
            idx, translated = await coro
            # Only record if it actually changed or is non-empty
            if translated and translated != texts[idx]:
                results[idx] = translated
        except Exception:
            # ignore individual failures
            pass

    return results

# Add near the top of the file (configurable)
MAX_TRANSLATABLE = 10000  # per-request hard limit; tune as needed

# Deduplicating batch translator (unchanged semantics)
async def batch_translate_texts_async(texts: List[str], target_lang: str) -> List[str]:
    """Batch translate multiple texts asynchronously with deduplication and optional retry.
    - Deduplicates identical source strings (preserves first-seen order)
    - Translates only unique strings, then maps results back to every original occurrence.
    - Enforces a hard per-request unique-text limit to avoid resource exhaustion.
    """
    if not texts:
        return []

    print(f"Translating {len(texts)} texts... (deduplicating first)")

    # Build mapping: unique_text -> list of original indices
    unique_map = {}
    unique_order = []
    for idx, t in enumerate(texts):
        key = t
        if key not in unique_map:
            unique_map[key] = []
            unique_order.append(key)
        unique_map[key].append(idx)

    unique_texts = unique_order  # ordered unique texts

    # Enforce limit on unique texts per request to avoid OOM/DoS from huge files
    if len(unique_texts) > MAX_TRANSLATABLE:
        # Fail-fast with informative error (client can choose to upload smaller chunks or use async job)
        raise HTTPException(
            status_code=413,
            detail=(
                f"Too many unique translatable strings ({len(unique_texts)}). "
                f"Please split the file or use the asynchronous large-file flow. "
                f"Limit is {MAX_TRANSLATABLE} unique strings per request."
            )
        )

    print(f"Reduced to {len(unique_texts)} unique texts for translation")

    # Now translate unique_texts using existing batching logic
    index_batches = create_optimized_batches_indices(unique_texts, max_chars=4500, max_texts=50)
    unique_translated = [None] * len(unique_texts)

    connector = aiohttp.TCPConnector(limit=40)
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        semaphore = asyncio.Semaphore(6)

        async def process_batch_with_semaphore(batch_indices):
            async with semaphore:
                batch_texts = [unique_texts[i] for i in batch_indices]
                return await process_batch_async(batch_texts, target_lang, session)

        tasks = [process_batch_with_semaphore(batch_indices) for batch_indices in index_batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Map batch results back to unique_translated
    for batch_indices, results in zip(index_batches, batch_results):
        if isinstance(results, Exception):
            for i in batch_indices:
                unique_translated[i] = unique_texts[i]
        else:
            for idx_in_batch, translated in zip(batch_indices, results):
                unique_translated[idx_in_batch] = clean_translation_text(translated)

    # Final fallback: fill any None with the original unique text
    for i in range(len(unique_translated)):
        if unique_translated[i] is None:
            unique_translated[i] = unique_texts[i]

    # Map back to full results: for each unique_text, set translated value at all original indices
    all_translated = [None] * len(texts)
    for u_idx, source_text in enumerate(unique_texts):
        translated_value = unique_translated[u_idx]
        for orig_idx in unique_map[source_text]:
            all_translated[orig_idx] = translated_value

    # Sanity fallback: ensure no None remain
    for i in range(len(all_translated)):
        if all_translated[i] is None:
            all_translated[i] = texts[i]

    return all_translated

def should_translate_text(text: str) -> bool:
    """Check if text should be translated"""
    if not text or not text.strip():
        return False

    if len(text.strip()) <= 1:
        return False

    if (text.strip().startswith('%') or
            text.strip().startswith('{') or
            text.strip().startswith('$')):
        return False

    if text.strip().isdigit() or len(text.strip()) == 1:
        return False

    return True


# File parsing functions
def parse_po_file(content: bytes) -> Tuple[List[str], any]:
    """Parse PO file and return translatable texts plus the POFile object"""
    po = polib.pofile(content.decode('utf-8'))
    texts = []
    # Store all entries in order, with their original index
    all_entries = []

    for entry in po:
        if entry.msgid and entry.msgid.strip() and not entry.obsolete:
            if should_translate_text(entry.msgid):
                texts.append(entry.msgid)
                all_entries.append(('translatable', entry, len(texts) - 1))
            else:
                all_entries.append(('non_translatable', entry, None))
        else:
            all_entries.append(('other', entry, None))

    return texts, (po, all_entries)

def parse_json_file(content: bytes) -> Tuple[List[str], any]:
    """Parse JSON file and find translatable strings"""
    try:
        data = json.loads(content.decode('utf-8'))
        texts = []
        structure = data

        def extract_strings(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    extract_strings(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_strings(item, f"{path}[{i}]")
            elif isinstance(obj, str) and should_translate_text(obj):
                texts.append(obj)

        extract_strings(data)
        return texts, structure
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")


def parse_csv_file(content: bytes) -> Tuple[List[str], any]:
    """Parse CSV file and find translatable strings"""
    try:
        content_str = content.decode('utf-8')
        reader = csv.reader(content_str.splitlines())
        texts = []
        rows = list(reader)

        for row in rows:
            for cell in row:
                if should_translate_text(cell):
                    texts.append(cell)

        return texts, rows
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {e}")


def parse_xliff_file(content: bytes) -> Tuple[List[str], any]:
    """Parse XLIFF file and extract translatable units"""
    try:
        root = ET.fromstring(content.decode('utf-8'))
        texts = []
        units = []

        # Namespace handling
        ns = {'xliff': 'urn:oasis:names:tc:xliff:document:1.2'}

        for trans_unit in root.findall('.//xliff:trans-unit', ns) or root.findall('.//trans-unit'):
            source = trans_unit.find('xliff:source', ns) or trans_unit.find('source')
            if source is not None and source.text and should_translate_text(source.text):
                texts.append(source.text)
                units.append(trans_unit)

        return texts, root
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid XLIFF file: {e}")


def parse_txt_file(content: bytes) -> Tuple[List[str], any]:
    """Parse text file line by line"""
    content_str = content.decode('utf-8')
    lines = content_str.split('\n')
    texts = []

    for line in lines:
        if should_translate_text(line.strip()):
            texts.append(line.strip())

    return texts, lines


def parse_properties_file(content: bytes) -> Tuple[List[str], any]:
    """Parse Java properties file"""
    content_str = content.decode('utf-8')
    lines = content_str.split('\n')
    texts = []
    properties = []

    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            if should_translate_text(value.strip()):
                texts.append(value.strip())
                properties.append((key.strip(), value.strip()))

    return texts, properties


# File writing functions
def write_translated_po_file(original_po_and_entries, translated_texts, output_path: str):
    """
    Write translated PO file maintaining exact entry order
    """
    po_obj, all_entries = original_po_and_entries
    translation_index = 0

    for entry_type, entry, text_index in all_entries:
        if entry_type == 'translatable' and text_index is not None:
            if translation_index < len(translated_texts) and translated_texts[translation_index] is not None:
                entry.msgstr = clean_translation_text(translated_texts[translation_index])
            else:
                entry.msgstr = ""  # Fallback to empty
            translation_index += 1
        else:
            # For non-translatable entries, ensure msgstr is empty if not already set
            if not entry.msgstr:
                entry.msgstr = ""

    po_obj.save(output_path)


def write_translated_json_file(original_structure, translated_texts, output_path: str):
    """Write translated JSON file"""
    translated_index = 0

    def replace_strings(obj):
        nonlocal translated_index
        if isinstance(obj, dict):
            return {k: replace_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_strings(item) for item in obj]
        elif isinstance(obj, str) and should_translate_text(obj):
            if translated_index < len(translated_texts):
                result = clean_translation_text(translated_texts[translated_index])
                translated_index += 1
                return result
        return obj

    translated_data = replace_strings(original_structure)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)


def write_translated_csv_file(original_rows, translated_texts, output_path: str):
    """Write translated CSV file"""
    translated_index = 0

    def should_translate_cell(cell):
        return cell and isinstance(cell, str) and should_translate_text(cell)

    translated_rows = []
    for row in original_rows:
        translated_row = []
        for cell in row:
            if should_translate_cell(cell):
                if translated_index < len(translated_texts):
                    translated_row.append(clean_translation_text(translated_texts[translated_index]))
                    translated_index += 1
                else:
                    translated_row.append(cell)
            else:
                translated_row.append(cell)
        translated_rows.append(translated_row)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(translated_rows)


def write_translated_xliff_file(original_root, translated_texts, output_path: str):
    """Write translated XLIFF file"""
    translated_index = 0
    ns = {'xliff': 'urn:oasis:names:tc:xliff:document:1.2'}

    for trans_unit in original_root.findall('.//xliff:trans-unit', ns) or original_root.findall('.//trans-unit'):
        source = trans_unit.find('xliff:source', ns) or trans_unit.find('source')
        if source is not None and source.text and should_translate_text(source.text):
            if translated_index < len(translated_texts):
                # Create or update target element
                target = trans_unit.find('xliff:target', ns) or trans_unit.find('target')
                if target is None:
                    target = ET.SubElement(trans_unit, 'target')
                target.text = clean_translation_text(translated_texts[translated_index])
                translated_index += 1

    tree = ET.ElementTree(original_root)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)


@app.get("/")
async def read_index():
    return FileResponse("templates/index.html")


@app.get("/supported-formats")
async def get_supported_formats():
    """Return supported file formats"""
    return {"formats": SUPPORTED_EXTENSIONS}


@app.post("/translate/")
async def translate_file(
        file: UploadFile = File(...),
        target_lang: str = Form(...),
        background_tasks: BackgroundTasks = None
):
    """Single endpoint that handles multiple file formats"""

    print(f"Received translation request for {file.filename} to language: {target_lang}")

    # Validate file type
    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
        )

    if target_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {target_lang}")

    file_id = str(uuid.uuid4())
    input_path = f"{file_id}_input{file_ext}"
    output_path = f"{file_id}_translated{file_ext}"

    try:
        # Save uploaded file
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        with open(input_path, "wb") as f:
            f.write(content)

        # Parse file based on type
        translatable_texts = []
        file_structure = None

        if file_ext == '.po':
            # Parse PO file
            translatable_texts, file_structure = parse_po_file(content)
            po_obj, all_entries = file_structure

            print(f"Found {len(translatable_texts)} translatable texts in {file.filename}")

            if not translatable_texts:
                raise HTTPException(status_code=400, detail="No translatable texts found in the file")

            # Create a direct mapping: each translatable text to its position in the translation list
            # No deduplication for PO files to maintain exact entry order
            start_time = time.time()

            # Mask texts for translation
            masked_texts = []
            placeholders_list = []
            for text in translatable_texts:
                masked, placeholders = mask_text(text)
                masked_texts.append(masked)
                placeholders_list.append(placeholders)

            # Translate all texts in order (no deduplication)
            translated_masked = await batch_translate_texts_async(masked_texts, target_lang)
            translation_time = time.time() - start_time
            print(f"Translation completed in {translation_time:.2f} seconds")

            # Unmask translated texts
            translated_texts = []
            for i, (masked_trans, placeholders) in enumerate(zip(translated_masked, placeholders_list)):
                translated_texts.append(unmask_text(masked_trans, placeholders))

            # Write PO file - this will use the exact order
            write_translated_po_file((po_obj, all_entries), translated_texts, output_path)

            # For preview, use the actual translated texts
            preview_source_texts = translatable_texts
            preview_translated_texts = translated_texts

        else:
            # Non-PO files (existing logic)
            if file_ext in ['.json']:
                translatable_texts, file_structure = parse_json_file(content)
            elif file_ext == '.csv':
                translatable_texts, file_structure = parse_csv_file(content)
            elif file_ext in ['.xlf', '.xliff']:
                translatable_texts, file_structure = parse_xliff_file(content)
            elif file_ext == '.txt':
                translatable_texts, file_structure = parse_txt_file(content)
            elif file_ext == '.properties':
                translatable_texts, file_structure = parse_properties_file(content)

            print(f"Found {len(translatable_texts)} translatable texts in {file.filename}")

            if not translatable_texts:
                raise HTTPException(status_code=400, detail="No translatable texts found in the file")

            # For non-PO types, use deduplication
            start_time = time.time()
            translated_texts = await batch_translate_texts_async(translatable_texts, target_lang)
            translation_time = time.time() - start_time

            print(f"Translation completed in {translation_time:.2f} seconds")

            # Write translated file based on type
            if file_ext in ['.json']:
                write_translated_json_file(file_structure, translated_texts, output_path)
            elif file_ext == '.csv':
                write_translated_csv_file(file_structure, translated_texts, output_path)
            elif file_ext in ['.xlf', '.xliff']:
                write_translated_xliff_file(file_structure, translated_texts, output_path)
            elif file_ext == '.txt':
                with open(output_path, 'w', encoding='utf-8') as f:
                    for line in file_structure:
                        f.write(line + '\n')
            elif file_ext == '.properties':
                with open(output_path, 'w', encoding='utf-8') as f:
                    for key, value in file_structure:
                        f.write(f"{key}={value}\n")

            preview_source_texts = translatable_texts
            preview_translated_texts = translated_texts

        # Prepare preview data
        translation_data = []
        translated_count = sum(1 for orig, trans in zip(preview_source_texts, preview_translated_texts)
                               if orig != trans and should_translate_text(orig))

        for idx, (original, translated) in enumerate(zip(preview_source_texts[:10], preview_translated_texts[:10])):
            clean_translated = clean_translation_text(translated)
            success = clean_translated != original and should_translate_text(original)

            translation_data.append({
                "original": original,
                "translated": clean_translated,
                "success": success,
                "reason": "Translated" if success else "Not translated" if not should_translate_text(
                    original) else "Translation failed"
            })

        # Clean up input file
        os.remove(input_path)

        # Return the translated file with preview data
        background_tasks.add_task(remove_file, output_path)

        response = FileResponse(
            output_path,
            filename=f"translated_{file.filename}",
            media_type='application/octet-stream'
        )

        # Add preview data as custom headers
        preview_data = {
            "translations": translation_data,
            "stats": {
                "total": len(preview_source_texts),
                "translated": translated_count,
                "skipped": len(preview_source_texts) - translated_count
            },
            "target_language": target_lang,
            "translation_time": f"{translation_time:.2f}s",
            "file_type": SUPPORTED_EXTENSIONS[file_ext]
        }

        response.headers["X-Translation-Preview"] = json.dumps(preview_data)
        response.headers["X-Translation-Stats"] = json.dumps(preview_data["stats"])

        return response

    except Exception as e:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        print(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


@app.get("/sitemap.xml", include_in_schema=False)
async def sitemap():
    """Dynamically generate sitemap for the website"""
    domain = "https://translatefiles.space"
    urls = [
        {"loc": f"{domain}/", "changefreq": "daily", "priority": "1.0"},
        {"loc": f"{domain}/supported-formats", "changefreq": "weekly", "priority": "0.8"},
        # You can add more pages here if needed
    ]

    sitemap_xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    sitemap_xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'

    for url in urls:
        sitemap_xml += "  <url>\n"
        sitemap_xml += f"    <loc>{url['loc']}</loc>\n"
        sitemap_xml += f"    <changefreq>{url['changefreq']}</changefreq>\n"
        sitemap_xml += f"    <priority>{url['priority']}</priority>\n"
        sitemap_xml += "  </url>\n"

    sitemap_xml += "</urlset>"

    return Response(content=sitemap_xml, media_type="application/xml")

# Helper function to load HTML pages
def read_html(file_name: str):
    file_path = os.path.join("templates", file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Legal and info pages
@app.get("/privacy-policy", response_class=HTMLResponse)
async def privacy_policy():
    return read_html("privacy-policy.html")

@app.get("/terms", response_class=HTMLResponse)
async def terms():
    return read_html("terms.html")

@app.get("/contact", response_class=HTMLResponse)
async def contact():
    return read_html("contact.html")

@app.get("/about", response_class=HTMLResponse)
async def about():
    return read_html("about.html")