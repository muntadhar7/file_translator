document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file');
    const fileDropZone = document.getElementById('fileDropZone');
    const fileName = document.getElementById('fileName');
    const form = document.getElementById('uploadForm');
    const translateBtn = document.getElementById('translateBtn');
    const messageDiv = document.getElementById('message');
    const downloadSection = document.getElementById('downloadSection');
    const downloadBtn = document.getElementById('downloadBtn');
    const previewContainer = document.getElementById('previewContainer');
    const previewCount = document.getElementById('previewCount');
    const previewTitle = document.getElementById('previewTitle');
    const statsSection = document.getElementById('statsSection');
    const statTranslated = document.getElementById('statTranslated');
    const statSkipped = document.getElementById('statSkipped');
    const statTotal = document.getElementById('statTotal');

    // Progress elements
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressPercentage = document.getElementById('progressPercentage');
    const progressText = document.getElementById('progressText');
    const progressMessage = document.getElementById('progressMessage');


    let translationData = [];
    let originalData = [];
    let progressInterval = null;

    // File drag and drop
    fileDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        fileDropZone.style.borderColor = 'var(--primary)';
        fileDropZone.style.background = 'rgba(124, 58, 237, 0.05)';
    });

    fileDropZone.addEventListener('dragleave', () => {
        fileDropZone.style.borderColor = 'var(--border)';
        fileDropZone.style.background = 'rgba(30, 41, 59, 0.5)';
    });

    fileDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            updateFileName();
        }
        fileDropZone.style.borderColor = 'var(--border)';
        fileDropZone.style.background = 'rgba(30, 41, 59, 0.5)';
    });

    fileInput.addEventListener('change', updateFileName);

    function updateFileName() {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            fileName.textContent = file.name;
            fileDropZone.classList.add('has-file');

            // Show file type info
            const fileExt = file.name.split('.').pop().toLowerCase();
            const supportedFormats = {
                'po': 'Gettext PO File',
                'json': 'JSON File',
                'csv': 'CSV File',
                'xlf': 'XLIFF File',
                'xliff': 'XLIFF File',
                'xml': 'XML File',
                'txt': 'Text File',
                'properties': 'Properties File'
            };

            if (supportedFormats[fileExt]) {
                fileName.textContent += ` (${supportedFormats[fileExt]})`;
            }

            previewFileContents(file);
        } else {
            fileName.textContent = '';
            fileDropZone.classList.remove('has-file');
        }
    }

    function previewFileContents(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const content = e.target.result;
            showFileInfo(file, content);
        };
        reader.readAsText(file);
    }

    function showFileInfo(file, content) {
        previewTitle.textContent = 'File Info';
        previewContainer.innerHTML = '';

        const fileExt = file.name.split('.').pop().toLowerCase();
        const fileSize = (file.size / 1024).toFixed(2);

        let fileType = 'Unknown';
        let sampleContent = '';

        switch(fileExt) {
            case 'po':
                fileType = 'Gettext PO File';
                sampleContent = extractPOContent(content);
                break;
            case 'json':
                fileType = 'JSON File';
                sampleContent = extractJSONContent(content);
                break;
            case 'csv':
                fileType = 'CSV File';
                sampleContent = extractCSVContent(content);
                break;
            case 'xlf':
            case 'xliff':
                fileType = 'XLIFF File';
                sampleContent = extractXLIFFContent(content);
                break;
            case 'xml':
                fileType = 'XML File';
                sampleContent = extractXMLContent(content);
                break;
            case 'txt':
                fileType = 'Text File';
                sampleContent = extractTXTContent(content);
                break;
            case 'properties':
                fileType = 'Properties File';
                sampleContent = extractPropertiesContent(content);
                break;
        }

        const infoHTML = `
            <div class="preview-item">
                <div class="preview-original">
                    <div class="preview-label">File Information</div>
                    <div class="preview-text">
                        <strong>Name:</strong> ${file.name}<br>
                        <strong>Type:</strong> ${fileType}<br>
                        <strong>Size:</strong> ${fileSize} KB<br>
                        <strong>Format:</strong> .${fileExt}
                    </div>
                </div>
            </div>
            ${sampleContent}
        `;

        previewContainer.innerHTML = infoHTML;
        previewCount.textContent = 'Ready to translate';
        statsSection.classList.add('hidden');
    }

    function extractPOContent(content) {
        const lines = content.split('\n');
        const entries = [];
        let currentEntry = {};

        for (const line of lines) {
            if (line.startsWith('msgid ')) {
                if (currentEntry.msgid) {
                    entries.push(currentEntry);
                }
                currentEntry = {
                    msgid: line.substring(6).replace(/^"|"$/g, ''),
                    msgstr: ''
                };
            } else if (line.startsWith('msgstr ') && currentEntry.msgid) {
                currentEntry.msgstr = line.substring(7).replace(/^"|"$/g, '');
            }
        }

        if (currentEntry.msgid) {
            entries.push(currentEntry);
        }

        const validEntries = entries.filter(entry => entry.msgid && entry.msgid.trim());

        if (validEntries.length === 0) {
            return '<div class="preview-item"><div class="preview-original"><div class="preview-label">Content</div><div class="preview-text">No translatable content found</div></div></div>';
        }

        let sampleHTML = '<div class="preview-item"><div class="preview-original"><div class="preview-label">Sample Content</div>';

        validEntries.slice(0, 3).forEach(entry => {
            sampleHTML += `<div class="preview-text" style="margin-bottom: 10px;"><strong>Original:</strong> ${escapeHtml(entry.msgid)}<br>`;
            if (entry.msgstr) {
                sampleHTML += `<strong>Current:</strong> ${escapeHtml(entry.msgstr)}`;
            }
            sampleHTML += '</div>';
        });

        if (validEntries.length > 3) {
            sampleHTML += `<div class="preview-text" style="color: var(--text-lighter);">... and ${validEntries.length - 3} more entries</div>`;
        }

        sampleHTML += '</div></div>';
        return sampleHTML;
    }

    function extractJSONContent(content) {
        try {
            const data = JSON.parse(content);
            const sample = JSON.stringify(data, null, 2).split('\n').slice(0, 10).join('\n');
            return `<div class="preview-item"><div class="preview-original"><div class="preview-label">JSON Structure</div><div class="preview-text"><pre style="white-space: pre-wrap; font-size: 0.9em;">${escapeHtml(sample)}</pre></div></div></div>`;
        } catch (e) {
            return '<div class="preview-item"><div class="preview-original"><div class="preview-label">Content</div><div class="preview-text">Invalid JSON file</div></div></div>';
        }
    }

    function extractCSVContent(content) {
        const lines = content.split('\n').slice(0, 5);
        return `<div class="preview-item"><div class="preview-original"><div class="preview-label">CSV Preview</div><div class="preview-text"><pre style="white-space: pre-wrap; font-size: 0.9em;">${escapeHtml(lines.join('\n'))}</pre></div></div></div>`;
    }

    function extractXLIFFContent(content) {
        return `<div class="preview-item"><div class="preview-original"><div class="preview-label">XLIFF File</div><div class="preview-text">XLIFF localization file ready for translation</div></div></div>`;
    }

    function extractXMLContent(content) {
        const lines = content.split('\n').slice(0, 5);
        return `<div class="preview-item"><div class="preview-original"><div class="preview-label">XML Preview</div><div class="preview-text"><pre style="white-space: pre-wrap; font-size: 0.9em;">${escapeHtml(lines.join('\n'))}</pre></div></div></div>`;
    }

    function extractTXTContent(content) {
        const lines = content.split('\n').slice(0, 5);
        return `<div class="preview-item"><div class="preview-original"><div class="preview-label">Text Preview</div><div class="preview-text"><pre style="white-space: pre-wrap;">${escapeHtml(lines.join('\n'))}</pre></div></div></div>`;
    }

    function extractPropertiesContent(content) {
        const lines = content.split('\n').filter(line => line.trim() && !line.startsWith('#')).slice(0, 5);
        return `<div class="preview-item"><div class="preview-original"><div class="preview-label">Properties Preview</div><div class="preview-text"><pre style="white-space: pre-wrap;">${escapeHtml(lines.join('\n'))}</pre></div></div></div>`;
    }

    function showTranslatedPreview(translations, stats, fileType) {
        previewTitle.textContent = `Translation Results - ${fileType}`;
        previewContainer.innerHTML = '';

        if (!translations || translations.length === 0) {
            previewContainer.innerHTML = `
                <div class="preview-placeholder">
                    <i class="fas fa-sync-alt"></i>
                    <p>No translation results to display</p>
                </div>
            `;
            previewCount.textContent = '0 items';
            return;
        }

        previewCount.textContent = `${translations.length} sample items`;

        translations.forEach(translation => {
            const item = document.createElement('div');
            item.className = 'preview-item fade-in';
            const statusClass = translation.success ? 'success' : 'error';
            const statusIcon = translation.success ? 'fa-check' : 'fa-times';
            const statusText = translation.success ? 'Successfully translated' : 'Translation failed';

            item.innerHTML = `
                <div class="preview-original">
                    <div class="preview-label">Original</div>
                    <div class="preview-text">${escapeHtml(translation.original)}</div>
                </div>
                <div class="preview-translated ${statusClass}">
                    <div class="preview-label">Translated</div>
                    <div class="preview-text">${escapeHtml(translation.translated)}</div>
                    <div class="preview-status" style="font-size: 0.7rem; margin-top: 5px; color: ${translation.success ? 'var(--success)' : 'var(--error)'}">
                        <i class="fas ${statusIcon}"></i>
                        ${statusText}
                    </div>
                </div>
            `;
            previewContainer.appendChild(item);
        });

        // Show stats
        if (stats) {
            updateStats(stats.total, stats.translated, stats.skipped);
        }
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Progress tracking functions
    function startProgressTracking(taskId) {
        // Clear any existing interval
        if (progressInterval) {
            clearInterval(progressInterval);
        }

        // Show progress container
        progressContainer.classList.add('progress-active');
        updateProgress(0, 0, 'Starting translation...');

        // Poll for progress updates
        progressInterval = setInterval(async () => {
            try {
                const response = await fetch(`/translation-progress/${taskId}`);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const progress = await response.json();

                if (progress.error) {
                    console.error('Progress error:', progress.error);
                    clearInterval(progressInterval);
                    showMessage(`Progress error: ${progress.error}`, 'error');
                    return;
                }

                updateProgress(
                    progress.current_batch || 0,
                    progress.total_batches || 1,
                    progress.message || 'Processing...'
                );

                // Stop polling if task is completed or failed
                if (progress.status === 'completed' || progress.status === 'error') {
                    clearInterval(progressInterval);
                    if (progress.status === 'completed') {
                        progressMessage.textContent = 'Translation completed!';
                    } else {
                        progressMessage.textContent = `Error: ${progress.message}`;
                        showMessage(`Translation error: ${progress.message}`, 'error');
                    }
                    // Hide progress after a delay
                    setTimeout(() => {
                        progressContainer.classList.remove('progress-active');
                    }, 2000);
                }
            } catch (error) {
                console.error('Error fetching progress:', error);
                // Don't show error to user for progress polling, just stop
                clearInterval(progressInterval);
            }
        }, 1000); // Poll every second
    }

    function updateProgress(completed, total, message) {
    const percentage = total > 0 ? Math.round((completed / total) * 100) : 0;

    progressBar.style.width = `${percentage}%`;
    progressPercentage.textContent = `${percentage}%`;

    // Just show the message without repeating batch numbers
    document.getElementById('batchInfo').textContent = message;
}

    async function waitForTranslationCompletion(taskId, maxAttempts = 300) {
        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            try {
                const response = await fetch(`/translation-result/${taskId}`);
                if (response.ok) {
                    const result = await response.json();
                    if (result.status === 'completed' || result.status === 'error') {
                        return result;
                    }
                }
                // Wait 1 second before checking again
                await new Promise(resolve => setTimeout(resolve, 1000));
            } catch (error) {
                console.error('Error checking translation result:', error);
                // Wait 1 second before retrying
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }
        throw new Error('Translation timeout - took too long to complete');
    }

    // Main form submission with progress tracking
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        if (!fileInput.files.length) {
            showMessage('Please select a file first.', 'error');
            return;
        }

        const formData = new FormData();
        const targetLang = document.getElementById('target_lang').value;

        formData.append('file', fileInput.files[0]);
        formData.append('target_lang', targetLang);

        // Show loading state
        translateBtn.innerHTML = '<div class="loading"></div> Translating...';
        translateBtn.disabled = true;
        messageDiv.innerHTML = '';
        downloadSection.classList.add('hidden');

        // Show progress container
        progressContainer.classList.add('progress-active');
        updateProgress(0, 0, 'Starting translation...');

        try {
            // First, start the translation task and get task ID
            const startResponse = await fetch('/start-translation/', {
                method: 'POST',
                body: formData
            });

            if (!startResponse.ok) {
                const errorData = await startResponse.json();
                throw new Error(errorData.detail || 'Failed to start translation');
            }

            const startData = await startResponse.json();
            const taskId = startData.task_id;

            // Start progress tracking
            startProgressTracking(taskId);

            // Wait for translation to complete
            const result = await waitForTranslationCompletion(taskId);

            if (result.status === 'completed') {
                // Download the translated file
                const downloadResponse = await fetch(`/download-translation/${taskId}`);
                if (downloadResponse.ok) {
                    const blob = await downloadResponse.blob();
                    const url = window.URL.createObjectURL(blob);
                    const filename = 'translated_' + fileInput.files[0].name;

                    // Set up download
                    downloadBtn.href = url;
                    downloadBtn.download = filename;
                    downloadSection.classList.remove('hidden');

                    // Show preview if available
                    if (result.preview) {
                        showTranslatedPreview(
                            result.preview.translations,
                            result.preview.stats,
                            result.preview.file_type
                        );

                        showMessage(
                            `Translation completed! ${result.preview.stats.translated}/${result.preview.stats.total} strings translated successfully.`,
                            'success'
                        );
                    } else {
                        showMessage('Translation completed successfully! Click download to get your file.', 'success');
                    }
                } else {
                    throw new Error('Failed to download translated file');
                }
            } else {
                throw new Error(result.message || 'Translation failed');
            }

        } catch (error) {
            showMessage(`Error: ${error.message}`, 'error');
            progressContainer.classList.remove('progress-active');
            if (progressInterval) {
                clearInterval(progressInterval);
            }
        } finally {
            translateBtn.innerHTML = '<i class="fas fa-magic"></i> Translate & Preview';
            translateBtn.disabled = false;
        }
    });

    function showMessage(text, type) {
        const icons = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            info: 'fas fa-info-circle'
        };

        messageDiv.innerHTML = `
            <div class="message ${type} fade-in">
                <i class="${icons[type]}"></i>
                <span>${text}</span>
            </div>
        `;

        // Auto-remove success messages after 5 seconds
        if (type === 'success') {
            setTimeout(() => {
                if (messageDiv.innerHTML.includes(text)) {
                    messageDiv.innerHTML = '';
                }
            }, 5000);
        }
    }

    function updateStats(total, translated, skipped) {
        statTotal.textContent = total;
        statTranslated.textContent = translated;
        statSkipped.textContent = skipped;
        statsSection.classList.remove('hidden');
    }
});