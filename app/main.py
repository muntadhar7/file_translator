import re
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
from typing import List, Dict, Tuple, Optional, Callable
import asyncio
import aiohttp
from fastapi.responses import Response, HTMLResponse

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

# Store progress in memory (use Redis in production)
translation_progress: Dict[str, dict] = {}

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


def create_progress_callback(task_id: str):
    """Create a progress callback function for a specific task"""

    def progress_callback(completed: int, total: int, message: str, processed_texts: int = 0, total_texts: int = 0):
        translation_progress[task_id] = {
            "status": "processing",
            "current_batch": completed,
            "total_batches": total,
            "progress": (completed / total * 100) if total > 0 else 0,
            "message": message,
            "processed_texts": processed_texts,
            "total_texts": total_texts
        }
        print(f"Progress: {completed}/{total} - {message}")  # Debug logging

    return progress_callback


def protect_specials(text: str) -> str:
    text = text.replace("%s", "__PERCENT_S__")
    text = text.replace("\n", "__NEWLINE__")
    text = re.sub(r"([*().])", r"__SYM_\1__", text)
    return text


def restore_specials(text: str) -> str:
    text = text.replace("__PERCENT_S__", "%s")
    text = text.replace("__NEWLINE__", "\n")
    text = re.sub(r"__SYM_(.)__", r"\1", text)
    return text


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


def parse_google_translate_response(response_data: List, original_texts: List[str]) -> List[str]:
    translations = [None] * len(original_texts)

    try:
        if response_data and isinstance(response_data, list):
            main_data = response_data[0]
            combined_text = ''.join(
                item[0] for item in main_data
                if item and isinstance(item, list) and isinstance(item[0], str)
            )

            # Extract using regex
            matches = re.findall(r"\[(\d+)](.*?)(?=\[\d+\]|$)", combined_text)
            for idx_str, text in matches:
                idx = int(idx_str)
                if 0 <= idx < len(original_texts):
                    translations[idx] = clean_translation_text(text)

    except Exception as e:
        print(f"Error parsing Google Translate response: {e}")

    # Fill in any missing translations with original
    return [t if t else original_texts[i] for i, t in enumerate(translations)]


async def translate_with_google_batch(texts: List[str], target_lang: str, session: aiohttp.ClientSession) -> List[str]:
    """Translate batch using Google Translate API"""
    try:

        tagged_texts = [f"[{i}]{text}" for i, text in enumerate(texts)]
        combined_text = " ".join(tagged_texts)

        params = {
            "client": "gtx",
            "dt": "t",
            "sl": "auto",
            "tl": target_lang,
            "q": combined_text
        }

        async with session.get(
                TRANSLATION_SERVICES["google_translate"]["url"],
                params=params,
                timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                data = await response.json()
                translated_texts = parse_google_translate_response(data, texts)
                return [clean_translation_text(text) for text in translated_texts]
            else:
                return texts
    except Exception as e:
        print(f"Google batch translation failed: {e}")
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
    """Translate single text using MyMemory"""
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

    # Final fallback: individual translations
    individual_results = []
    for text in batch:
        translated = translate_with_mymemory_single(text, target_lang)
        individual_results.append(translated)
        await asyncio.sleep(0.1)

    return individual_results


async def batch_translate_texts_async(
        texts: List[str],
        target_lang: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> List[str]:
    """Batch translate multiple texts asynchronously with progress tracking"""
    if not texts:
        return []

    print(f"Translating {len(texts)} texts...")

    batches = create_optimized_batches(texts, max_chars=4500, max_texts=50)
    all_translated = [None] * len(texts)
    text_to_index = {}

    current_idx = 0
    for batch_idx, batch in enumerate(batches):
        for text in batch:
            text_to_index[text] = current_idx
            current_idx += 1

    # Initialize progress
    total_batches = len(batches)
    completed_batches = 0

    if progress_callback:
        progress_callback(0, total_batches, "Starting translation...", 0, len(texts))

    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(3)

        async def process_batch_with_semaphore(batch, batch_idx):
            async with semaphore:
                result = await process_batch_async(batch, target_lang, session)

                # Update progress
                nonlocal completed_batches
                completed_batches += 1
                if progress_callback:
                    processed_texts = sum(1 for t in all_translated if t is not None)
                    progress_callback(completed_batches, total_batches,
                                      f"Processed batch {batch_idx + 1}/{total_batches}",
                                      processed_texts, len(texts))

                return result

        tasks = []
        for batch_idx, batch in enumerate(batches):
            task = process_batch_with_semaphore(batch, batch_idx)
            tasks.append(task)

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    successful_translations = 0
    fallback_translations = 0

    for batch_idx, (batch, results) in enumerate(zip(batches, batch_results)):
        if isinstance(results, Exception):
            for text in batch:
                original_idx = text_to_index[text]
                all_translated[original_idx] = clean_translation_text(
                    translate_with_mymemory_single(text, target_lang)
                )
                fallback_translations += 1
        else:
            for text, translated in zip(batch, results):
                original_idx = text_to_index[text]
                all_translated[original_idx] = clean_translation_text(translated)
                successful_translations += 1

    # Handle any remaining None values
    for i in range(len(all_translated)):
        if all_translated[i] is None:
            all_translated[i] = texts[i]

    if progress_callback:
        progress_callback(total_batches, total_batches,
                          f"Completed! Success: {successful_translations}, Fallback: {fallback_translations}",
                          len(texts), len(texts))

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
    """Parse PO file and return translatable texts"""
    po = polib.pofile(content.decode('utf-8'))
    texts = []
    entries = []

    for entry in po:
        if entry.msgid and entry.msgid.strip() and not entry.obsolete:
            if should_translate_text(entry.msgid):
                texts.append(entry.msgid)
                entries.append(entry)

    return texts, entries


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
        root = ET.fromstring(content)
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
def write_translated_po_file(original_entries, translated_texts, output_path: str):
    """Write translated PO file"""
    po = polib.POFile()

    for i, (entry, translated) in enumerate(zip(original_entries, translated_texts)):
        new_entry = polib.POEntry(
            msgid=entry.msgid,
            msgstr=clean_translation_text(translated)
        )
        po.append(new_entry)

    po.save(output_path)


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


# Progress Tracking Endpoints
@app.post("/start-translation/")
async def start_translation(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        target_lang: str = Form(...)
):
    """Start translation with progress tracking"""
    task_id = str(uuid.uuid4())

    # Initialize progress
    translation_progress[task_id] = {
        "status": "starting",
        "current_batch": 0,
        "total_batches": 0,
        "progress": 0,
        "message": "Initializing translation...",
        "processed_texts": 0,
        "total_texts": 0
    }

    # Create progress callback
    progress_callback = create_progress_callback(task_id)

    # Start translation in background
    background_tasks.add_task(
        run_translation_with_progress,
        task_id,
        file,
        target_lang,
        progress_callback
    )

    return {"task_id": task_id, "status": "started"}


@app.get("/translation-progress/{task_id}")
async def get_translation_progress(task_id: str):
    """Get current progress for a translation task"""
    if task_id not in translation_progress:
        return JSONResponse(
            status_code=404,
            content={"error": "Task not found"}
        )
    return translation_progress[task_id]


@app.get("/translation-result/{task_id}")
async def get_translation_result(task_id: str):
    """Get final translation result"""
    if task_id not in translation_progress:
        return JSONResponse(
            status_code=404,
            content={"error": "Task not found"}
        )

    progress = translation_progress[task_id]
    if progress["status"] not in ["completed", "error"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Translation not completed yet"}
        )

    return {
        "status": progress["status"],
        "result": progress.get("result"),
        "preview": progress.get("preview"),
        "stats": progress.get("stats")
    }


async def run_translation_with_progress(
        task_id: str,
        file: UploadFile,
        target_lang: str,
        progress_callback: Optional[Callable] = None
):
    """Run translation with progress updates"""
    try:
        # Validate file type
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in SUPPORTED_EXTENSIONS:
            translation_progress[task_id].update({
                "status": "error",
                "message": f"Unsupported file type: {file_ext}"
            })
            return

        if target_lang not in SUPPORTED_LANGUAGES:
            translation_progress[task_id].update({
                "status": "error",
                "message": f"Unsupported language: {target_lang}"
            })
            return

        file_id = str(uuid.uuid4())
        input_path = f"{file_id}_input{file_ext}"
        output_path = f"{file_id}_translated{file_ext}"

        # Save uploaded file
        content = await file.read()
        if not content:
            translation_progress[task_id].update({
                "status": "error",
                "message": "Empty file"
            })
            return

        with open(input_path, "wb") as f:
            f.write(content)

        # Parse file based on type
        translatable_texts = []
        file_structure = None

        if progress_callback:
            progress_callback(0, 1, "Parsing file...", 0, 0)

        if file_ext == '.po':
            translatable_texts, file_structure = parse_po_file(content)
        elif file_ext in ['.json']:
            translatable_texts, file_structure = parse_json_file(content)
        elif file_ext == '.csv':
            translatable_texts, file_structure = parse_csv_file(content)
        elif file_ext in ['.xlf', '.xliff']:
            translatable_texts, file_structure = parse_xliff_file(content)
        elif file_ext == '.txt':
            translatable_texts, file_structure = parse_txt_file(content)
        elif file_ext == '.properties':
            translatable_texts, file_structure = parse_properties_file(content)

        if progress_callback:
            progress_callback(1, 1, f"Found {len(translatable_texts)} translatable texts", 0, len(translatable_texts))

        if not translatable_texts:
            translation_progress[task_id].update({
                "status": "error",
                "message": "No translatable texts found in the file"
            })
            os.remove(input_path)
            return

        protected = [protect_specials(t) for t in translatable_texts]

        # Batch translate all texts
        start_time = time.time()
        translated_texts = await batch_translate_texts_async(protected, target_lang, progress_callback)
        translation_time = time.time() - start_time

        print(f"Translation completed in {translation_time:.2f} seconds")

        # Write translated file based on type
        translated_texts = [restore_specials(t) for t in translated_texts]

        if file_ext == '.po':
            write_translated_po_file(file_structure, translated_texts, output_path)
        elif file_ext in ['.json']:
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

        # Prepare preview data
        translation_data = []
        translated_count = sum(1 for orig, trans in zip(translatable_texts, translated_texts) if orig != trans)

        for idx, (original, translated) in enumerate(zip(translatable_texts[:10], translated_texts[:10])):
            clean_translated = clean_translation_text(translated)
            success = clean_translated != original

            translation_data.append({
                "original": original,
                "translated": clean_translated,
                "success": success,
                "reason": "Translated" if success else "Translation failed"
            })

        # Clean up input file
        os.remove(input_path)

        # Store result for download
        translation_progress[task_id].update({
            "status": "completed",
            "progress": 100,
            "message": "Translation completed successfully",
            "result": output_path,
            "preview": {
                "translations": translation_data,
                "stats": {
                    "total": len(translatable_texts),
                    "translated": translated_count,
                    "skipped": len(translatable_texts) - translated_count
                },
                "target_language": target_lang,
                "translation_time": f"{translation_time:.2f}s",
                "file_type": SUPPORTED_EXTENSIONS[file_ext]
            },
            "stats": {
                "total": len(translatable_texts),
                "translated": translated_count,
                "skipped": len(translatable_texts) - translated_count
            }
        })

    except Exception as e:
        print(f"Translation error: {e}")
        translation_progress[task_id].update({
            "status": "error",
            "message": f"Translation failed: {str(e)}"
        })


@app.get("/download-translation/{task_id}")
async def download_translation(task_id: str):
    """Download the translated file"""
    if task_id not in translation_progress:
        raise HTTPException(status_code=404, detail="Task not found")

    progress = translation_progress[task_id]
    if progress["status"] != "completed":
        raise HTTPException(status_code=400, detail="Translation not completed")

    output_path = progress.get("result")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Translated file not found")

    return FileResponse(
        output_path,
        filename=f"translated_file{os.path.splitext(output_path)[1]}",
        media_type='application/octet-stream'
    )


# Keep your original endpoint for backward compatibility
@app.post("/translate/")
async def translate_file(
        file: UploadFile = File(...),
        target_lang: str = Form(...),
        background_tasks: BackgroundTasks = None
):
    """Single endpoint that handles multiple file formats (original version)"""
    # ... keep your original /translate/ endpoint code exactly as it was ...
    # This ensures your current frontend still works


@app.get("/")
async def read_index():
    return FileResponse("templates/index.html")




# Helper function to load HTML/XML files
def read_html(file_name: str):
    file_path = os.path.join("templates", file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/supported-formats")
async def get_supported_formats():
    """Return supported file formats"""
    return {"formats": SUPPORTED_EXTENSIONS}

@app.get("/supported_formats", response_class=HTMLResponse)
async def supported_formats():
    return read_html("supported-formats.html")

@app.get("/sitemap.xml", include_in_schema=False)
async def sitemap():
    content = read_html("sitemap.xml")
    return Response(content=content, media_type="application/xml")


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



@app.get("/robots.txt", include_in_schema=False)
async def robots_txt():
    content = read_html("robots.txt")
    return Response(content=content, media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
