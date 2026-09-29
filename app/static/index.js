/**
 * RAG PDF Chatbot Client Application (Single Page App)
 * Focuses on zero-dependency, semantic DOM manipulation, security hardening (XSS mitigation),
 * and clean user experience with real-time feedback.
 */

(function () {
    'use strict';

    // --- State management ---
    const state = {
        apiKey: '',
        storageType: 'session', // 'session' or 'local'
        currentSessionId: null,
        sessions: [],
        sessionsPage: 1,
        sessionsLimit: 8,
        sessionsTotal: 0,
        documents: [],
        docsPage: 1,
        docsLimit: 5,
        docsTotalCount: 0, // inferred by checking return length vs limit
        activeChatController: null,
        activeUploadXHR: null
    };

    // --- DOM Elements Cache ---
    const elements = {
        serverStatus: document.getElementById('server-status'),
        statusText: document.getElementById('status-text'),
        btnOpenDocs: document.getElementById('btn-open-docs'),
        btnOpenSettings: document.getElementById('btn-open-settings'),
        btnNewChat: document.getElementById('btn-new-chat'),
        sessionsContainer: document.getElementById('sessions-container'),
        btnPrevSessions: document.getElementById('btn-prev-sessions'),
        btnNextSessions: document.getElementById('btn-next-sessions'),
        sessionsPageIndicator: document.getElementById('sessions-page-indicator'),
        chatMessages: document.getElementById('chat-messages'),
        chatForm: document.getElementById('chat-form'),
        promptInput: document.getElementById('prompt-input'),
        charCounter: document.getElementById('char-counter'),
        btnSendMessage: document.getElementById('btn-send-message'),
        
        // Settings Modal
        settingsModal: document.getElementById('settings-modal'),
        btnCloseSettings: document.getElementById('btn-close-settings'),
        settingsForm: document.getElementById('settings-form'),
        inputApiKey: document.getElementById('input-api-key'),
        btnToggleKeyVisibility: document.getElementById('btn-toggle-key-visibility'),
        btnTestConnection: document.getElementById('btn-test-connection'),
        btnForgetKey: document.getElementById('btn-forget-key'),
        welcomeSetupKey: document.getElementById('welcome-setup-key'),
        welcomeUploadPdf: document.getElementById('welcome-upload-pdf'),
        
        // Docs Modal
        docsModal: document.getElementById('docs-modal'),
        btnCloseDocs: document.getElementById('btn-close-docs'),
        dragDropZone: document.getElementById('drag-drop-zone'),
        fileInput: document.getElementById('file-input'),
        uploadProgressContainer: document.getElementById('upload-progress-container'),
        uploadingFilename: document.getElementById('uploading-filename'),
        uploadPercentage: document.getElementById('upload-percentage'),
        uploadProgressBar: document.getElementById('upload-progress-bar'),
        uploadStatusDetails: document.getElementById('upload-status-details'),
        docsTableBody: document.getElementById('docs-table-body'),
        btnPrevDocs: document.getElementById('btn-prev-docs'),
        btnNextDocs: document.getElementById('btn-next-docs'),
        docsPageIndicator: document.getElementById('docs-page-indicator'),
        
        // Toasts
        toastContainer: document.getElementById('toast-container')
    };

    // --- Core API Helpers ---
    function getHeaders() {
        const headers = {
            'Content-Type': 'application/json'
        };
        if (state.apiKey) {
            headers['X-API-Key'] = state.apiKey;
        }
        return headers;
    }

    async function apiRequest(url, options = {}) {
        options.headers = { ...getHeaders(), ...options.headers };
        try {
            const response = await fetch(url, options);
            if (!response.ok) {
                let detail = 'An unexpected error occurred.';
                try {
                    const errJson = await response.json();
                    if (Array.isArray(errJson.detail)) {
                        detail = errJson.detail.map(issue => {
                            const field = (issue.loc || []).join('.');
                            return `${field ? `${field}: ` : ''}${issue.msg || 'Invalid value'}`;
                        }).join('; ');
                    } else if (typeof errJson.detail === 'string') {
                        detail = errJson.detail;
                    }
                } catch (_) {}
                
                const err = new Error(detail);
                err.status = response.status;
                throw err;
            }
            return await response.json();
        } catch (error) {
            if (error.name === 'AbortError') {
                throw error;
            }
            loggerError(error.message);
            throw error;
        }
    }

    // --- Toast / Notification Manager ---
    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type === 'error' ? 'error' : type === 'success' ? 'success' : ''}`;
        toast.textContent = message;
        elements.toastContainer.appendChild(toast);
        
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    function loggerError(msg) {
        console.error('[RAG Client Error]:', msg);
    }

    // --- Initialization & API Key Management ---
    function loadStoredApiKey() {
        // Try session storage first
        let key = sessionStorage.getItem('rag_api_key');
        let type = 'session';
        
        if (!key || !key.trim()) {
            // Fallback to local storage
            key = localStorage.getItem('rag_api_key');
            type = 'local';
        }

        // Auto-fill default local development key if none or empty
        if (!key || !key.trim()) {
            key = 'change_me_in_production';
            type = 'local';
            try { localStorage.setItem('rag_api_key', key); } catch (_) {}
        }
        
        if (key) {
            state.apiKey = key;
            state.storageType = type;
            elements.inputApiKey.value = key;
            const storageRadio = document.querySelector(`input[name="storage-type"][value="${type}"]`);
            if (storageRadio) storageRadio.checked = true;
            enableChatComposer(true);
        } else {
            enableChatComposer(false);
        }
    }

    function saveApiKey(key, strategy) {
        state.apiKey = key;
        state.storageType = strategy;
        
        // Clean both first
        localStorage.removeItem('rag_api_key');
        sessionStorage.removeItem('rag_api_key');
        
        if (key) {
            if (strategy === 'local') {
                localStorage.setItem('rag_api_key', key);
            } else {
                sessionStorage.setItem('rag_api_key', key);
            }
            enableChatComposer(true);
            showToast('API Key saved successfully!', 'success');
        } else {
            enableChatComposer(false);
        }
    }

    function forgetApiKey() {
        localStorage.removeItem('rag_api_key');
        sessionStorage.removeItem('rag_api_key');
        state.apiKey = '';
        elements.inputApiKey.value = '';
        enableChatComposer(false);
        showToast('API Key cleared from storage.', 'info');
        renderWelcomeScreen();
        renderSessionsPlaceholder('No API key configured.');
    }

    function enableChatComposer(enabled) {
        elements.promptInput.disabled = !enabled;
        elements.btnSendMessage.disabled = !enabled;
        if (!enabled) {
            elements.promptInput.placeholder = 'Please configure API Settings with a valid X-API-Key to chat...';
        } else {
            elements.promptInput.placeholder = 'Ask a question about your uploaded PDF documents... (Enter to send, Shift+Enter for new line)';
        }
    }

    // --- Health / Status Checker ---
    async function checkServerStatus() {
        try {
            const data = await apiRequest('/health');
            if (data && data.status === 'healthy') {
                // Now check readiness
                const readyRes = await fetch('/health/ready');
                const readyData = await readyRes.json();
                
                elements.serverStatus.className = 'status-indicator ready';
                if (readyData.status === 'ready') {
                    elements.statusText.textContent = 'Server: Ready';
                } else if (readyData.components && readyData.components.vector_store === 'not_initialized') {
                    elements.serverStatus.className = 'status-indicator not-ready';
                    elements.statusText.textContent = 'Index Missing';
                } else {
                    elements.serverStatus.className = 'status-indicator not-ready';
                    elements.statusText.textContent = 'Server: Not Ready';
                }
            } else {
                elements.serverStatus.className = 'status-indicator error';
                elements.statusText.textContent = 'Server: Unhealthy';
            }
        } catch (error) {
            elements.serverStatus.className = 'status-indicator error';
            elements.statusText.textContent = 'Server Offline';
        }
    }

    // --- Modal Navigation ---
    function openModal(modal) {
        modal.classList.add('active');
        modal.setAttribute('aria-hidden', 'false');
        
        // Trap focus
        const focusable = modal.querySelectorAll('button, [href], input, select, textarea, [tabindex="0"]');
        if (focusable.length > 0) {
            focusable[0].focus();
        }
        
        // Escape handler
        const escListener = function(e) {
            if (e.key === 'Escape') {
                closeModal(modal);
                document.removeEventListener('keydown', escListener);
            }
        };
        document.addEventListener('keydown', escListener);
    }

    function closeModal(modal) {
        modal.classList.remove('active');
        modal.setAttribute('aria-hidden', 'true');
    }

    // --- Document Manager Actions ---
    async function loadDocuments() {
        if (!state.apiKey) {
            renderDocsTablePlaceholder('Please configure an API Key first.');
            return;
        }
        
        renderDocsTablePlaceholder('Loading documents...');
        
        const offset = (state.docsPage - 1) * state.docsLimit;
        try {
            const docs = await apiRequest(`/api/v1/documents?limit=${state.docsLimit}&offset=${offset}`);
            state.documents = docs;
            renderDocumentsTable(docs);
        } catch (error) {
            if (error.status === 401 || error.status === 403) {
                renderDocsTablePlaceholder('Authentication failed. Check API Key.');
            } else {
                renderDocsTablePlaceholder(`Error: ${error.message}`);
            }
        }
    }

    function renderDocsTablePlaceholder(msg) {
        elements.docsTableBody.innerHTML = '';
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.colSpan = 6;
        td.className = 'table-empty';
        td.textContent = msg;
        tr.appendChild(td);
        elements.docsTableBody.appendChild(tr);
        elements.btnPrevDocs.disabled = true;
        elements.btnNextDocs.disabled = true;
    }

    function formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function renderDocumentsTable(docs) {
        elements.docsTableBody.innerHTML = '';
        
        if (docs.length === 0) {
            renderDocsTablePlaceholder('No documents uploaded yet.');
            elements.btnPrevDocs.disabled = state.docsPage <= 1;
            elements.btnNextDocs.disabled = true;
            return;
        }
        
        docs.forEach(doc => {
            const tr = document.createElement('tr');
            
            // Filename (safely escaped)
            const tdName = document.createElement('td');
            tdName.textContent = doc.filename;
            tdName.style.wordBreak = 'break-all';
            
            // Size
            const tdSize = document.createElement('td');
            tdSize.textContent = formatBytes(doc.file_size_bytes || 0);
            
            // Chunks
            const tdChunks = document.createElement('td');
            tdChunks.textContent = doc.chunk_count || 0;
            
            // Date
            const tdDate = document.createElement('td');
            tdDate.textContent = doc.created_at ? new Date(doc.created_at).toLocaleString() : 'N/A';
            
            // Status Badge
            const tdStatus = document.createElement('td');
            const badge = document.createElement('span');
            const statusVal = (doc.status || '').toLowerCase();
            badge.className = `badge ${statusVal}`;
            badge.textContent = statusVal || 'unknown';
            tdStatus.appendChild(badge);
            
            // Actions
            const tdActions = document.createElement('td');
            const btnDelete = document.createElement('button');
            btnDelete.className = 'btn btn-sm btn-danger';
            btnDelete.textContent = 'Delete';
            btnDelete.addEventListener('click', () => confirmDeleteDocument(doc.document_id, doc.filename));
            tdActions.appendChild(btnDelete);
            
            tr.appendChild(tdName);
            tr.appendChild(tdSize);
            tr.appendChild(tdChunks);
            tr.appendChild(tdDate);
            tr.appendChild(tdStatus);
            tr.appendChild(tdActions);
            
            elements.docsTableBody.appendChild(tr);
        });

        // Pagination buttons
        elements.btnPrevDocs.disabled = state.docsPage <= 1;
        // Inferred Next Page check: if list count matches the requested limit, there is probably a next page
        elements.btnNextDocs.disabled = docs.length < state.docsLimit;
        elements.docsPageIndicator.textContent = `Page ${state.docsPage}`;
    }

    async function confirmDeleteDocument(docId, filename) {
        if (!confirm(`Are you sure you want to delete "${filename}"? This will remove it from search index entirely.`)) {
            return;
        }
        
        try {
            await apiRequest(`/api/v1/documents/${docId}`, { method: 'DELETE' });
            showToast('Document deleted successfully', 'success');
            // Reset to page 1 to reload
            state.docsPage = 1;
            loadDocuments();
            checkServerStatus(); // status could change from Ready to Not Ready if index cleared
        } catch (error) {
            showToast(`Delete failed: ${error.message}`, 'error');
        }
    }

    // --- XMLHttpRequest Drag & Drop PDF Upload Ingestion ---
    function uploadFile(file) {
        if (!state.apiKey) {
            showToast('Please configure your API settings first!', 'error');
            return;
        }

        if (file.size > 50 * 1024 * 1024) {
            showToast('File too large. Maximum size allowed is 50 MB.', 'error');
            return;
        }

        if (!file.name.toLowerCase().endsWith('.pdf')) {
            showToast('Invalid file format. Only PDF files are supported.', 'error');
            return;
        }

        // Setup XHR and show upload UI
        elements.uploadProgressContainer.classList.remove('hidden');
        elements.uploadingFilename.textContent = file.name;
        elements.uploadProgressBar.style.width = '0%';
        elements.uploadPercentage.textContent = '0%';
        elements.uploadStatusDetails.textContent = 'Preparing file transfer...';
        elements.uploadStatusDetails.className = 'status-details';

        const formData = new FormData();
        formData.append('file', file);

        const xhr = new XMLHttpRequest();
        state.activeUploadXHR = xhr;

        // Hook progress events
        xhr.upload.addEventListener('progress', function (e) {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                elements.uploadProgressBar.style.width = `${percent}%`;
                elements.uploadPercentage.textContent = `${percent}%`;
                if (percent === 100) {
                    elements.uploadStatusDetails.textContent = 'Uploading complete. Server is processing and indexing text...';
                    elements.uploadStatusDetails.classList.add('pulse');
                } else {
                    elements.uploadStatusDetails.textContent = `Uploading chunk... (${formatBytes(e.loaded)} / ${formatBytes(e.total)})`;
                }
            }
        });

        xhr.addEventListener('load', function () {
            state.activeUploadXHR = null;
            elements.uploadStatusDetails.classList.remove('pulse');
            
            if (xhr.status === 201) {
                elements.uploadStatusDetails.textContent = 'Successfully processed and indexed!';
                elements.uploadStatusDetails.className = 'status-details text-success';
                showToast('Ingestion completed successfully!', 'success');
                setTimeout(() => {
                    elements.uploadProgressContainer.classList.add('hidden');
                }, 2000);
                
                state.docsPage = 1;
                loadDocuments();
                checkServerStatus(); // should trigger active ready status
            } else {
                let detail = 'Server failed to process document.';
                try {
                    const resp = JSON.parse(xhr.responseText);
                    detail = resp.detail || detail;
                } catch (_) {}
                handleUploadFailure(detail);
            }
        });

        xhr.addEventListener('error', function () {
            state.activeUploadXHR = null;
            handleUploadFailure('Network connection failed during upload.');
        });

        xhr.addEventListener('abort', function () {
            state.activeUploadXHR = null;
            handleUploadFailure('Upload cancelled by user.');
        });

        xhr.open('POST', '/api/v1/documents/upload');
        xhr.setRequestHeader('X-API-Key', state.apiKey);
        xhr.send(formData);
    }

    function handleUploadFailure(msg) {
        elements.uploadProgressBar.style.width = '0%';
        elements.uploadPercentage.textContent = 'Error';
        elements.uploadStatusDetails.textContent = msg;
        elements.uploadStatusDetails.className = 'status-details text-danger';
        elements.uploadStatusDetails.classList.remove('pulse');
        showToast(`Upload failed: ${msg}`, 'error');
    }

    // --- Chat Sessions Management ---
    async function loadSessions() {
        if (!state.apiKey) {
            renderSessionsPlaceholder('Please configure API Key first.');
            return;
        }

        renderSessionsPlaceholder('Loading sessions...');

        const offset = (state.sessionsPage - 1) * state.sessionsLimit;
        try {
            const data = await apiRequest(`/api/v1/sessions?limit=${state.sessionsLimit}&offset=${offset}`);
            state.sessions = data.sessions || [];
            state.sessionsTotal = data.total || 0;
            renderSessionsList(state.sessions);
        } catch (error) {
            if (error.status === 401 || error.status === 403) {
                renderSessionsPlaceholder('Auth failed. Check key.');
            } else {
                renderSessionsPlaceholder(`Error: ${error.message}`);
            }
        }
    }

    function renderSessionsPlaceholder(msg) {
        elements.sessionsContainer.innerHTML = '';
        const div = document.createElement('div');
        div.className = 'list-placeholder';
        div.textContent = msg;
        elements.sessionsContainer.appendChild(div);
        elements.btnPrevSessions.disabled = true;
        elements.btnNextSessions.disabled = true;
    }

    function renderSessionsList(sessions) {
        elements.sessionsContainer.innerHTML = '';
        
        if (sessions.length === 0) {
            renderSessionsPlaceholder('No active sessions. Start a new one!');
            elements.btnPrevSessions.disabled = state.sessionsPage <= 1;
            elements.btnNextSessions.disabled = true;
            return;
        }

        sessions.forEach(session => {
            const divItem = document.createElement('div');
            divItem.className = `session-item ${state.currentSessionId === session.session_id ? 'active' : ''}`;
            divItem.style.cursor = 'pointer';
            
            // Switch session on click of body (avoid buttons)
            divItem.addEventListener('click', (e) => {
                if (e.target.closest('.btn-delete-session')) return;
                switchSession(session.session_id);
            });

            const divInfo = document.createElement('div');
            divInfo.className = 'session-info';

            const spanTitle = document.createElement('span');
            spanTitle.className = 'session-title';
            spanTitle.textContent = `Session: ${session.session_id.substring(0, 8)}...`;

            const spanMeta = document.createElement('span');
            spanMeta.className = 'session-meta';
            const count = session.message_count || 0;
            spanMeta.textContent = `${count} message${count !== 1 ? 's' : ''}`;

            divInfo.appendChild(spanTitle);
            divInfo.appendChild(spanMeta);

            const btnDelete = document.createElement('button');
            btnDelete.className = 'btn-delete-session';
            btnDelete.innerHTML = '&times;';
            btnDelete.title = 'Delete Session';
            btnDelete.addEventListener('click', (e) => {
                e.stopPropagation();
                confirmDeleteSession(session.session_id);
            });

            divItem.appendChild(divInfo);
            divItem.appendChild(btnDelete);
            
            elements.sessionsContainer.appendChild(divItem);
        });

        // Pagination buttons
        elements.btnPrevSessions.disabled = state.sessionsPage <= 1;
        elements.btnNextSessions.disabled = (state.sessionsPage * state.sessionsLimit) >= state.sessionsTotal;
        elements.sessionsPageIndicator.textContent = `Page ${state.sessionsPage}`;
    }

    async function switchSession(sessionId) {
        state.currentSessionId = sessionId;
        
        // Highlight active sidebar item
        const items = elements.sessionsContainer.querySelectorAll('.session-item');
        items.forEach((item, idx) => {
            const sess = state.sessions[idx];
            if (sess && sess.session_id === sessionId) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });

        // Load messages
        elements.chatMessages.innerHTML = '';
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'list-placeholder';
        loadingDiv.textContent = 'Loading message history...';
        elements.chatMessages.appendChild(loadingDiv);

        try {
            const data = await apiRequest(`/api/v1/sessions/${sessionId}`);
            renderSessionHistory(data.messages || []);
        } catch (error) {
            elements.chatMessages.innerHTML = '';
            const errDiv = document.createElement('div');
            errDiv.className = 'list-placeholder text-danger';
            errDiv.textContent = `Failed to load history: ${error.message}`;
            elements.chatMessages.appendChild(errDiv);
        }
    }

    async function confirmDeleteSession(sessionId) {
        if (!confirm('Are you sure you want to delete this session and its message history?')) {
            return;
        }

        try {
            await apiRequest(`/api/v1/sessions/${sessionId}`, { method: 'DELETE' });
            showToast('Session deleted', 'success');
            
            if (state.currentSessionId === sessionId) {
                state.currentSessionId = null;
                renderWelcomeScreen();
            }
            
            state.sessionsPage = 1;
            loadSessions();
        } catch (error) {
            showToast(`Delete failed: ${error.message}`, 'error');
        }
    }

    // --- Message List Rendering ---
    function renderWelcomeScreen() {
        elements.chatMessages.innerHTML = '';
        
        const welcome = document.createElement('div');
        welcome.className = 'welcome-container';
        
        const h2 = document.createElement('h2');
        const p = document.createElement('p');
        const actions = document.createElement('div');
        actions.className = 'setup-quick-actions';

        if (state.apiKey) {
            h2.textContent = 'Ready to Chat';
            p.textContent = 'Your documents are indexed and ready. Ask any question in the prompt box below!';

            const btnPdf = document.createElement('button');
            btnPdf.className = 'btn btn-primary';
            btnPdf.textContent = 'Manage Documents';
            btnPdf.addEventListener('click', () => openModal(elements.docsModal));

            const btnSettings = document.createElement('button');
            btnSettings.className = 'btn btn-secondary';
            btnSettings.textContent = 'API Settings';
            btnSettings.addEventListener('click', () => openModal(elements.settingsModal));

            actions.appendChild(btnPdf);
            actions.appendChild(btnSettings);
        } else {
            h2.textContent = 'Welcome to RAG PDF Chatbot';
            p.textContent = 'Configure your API Key and upload PDF documents to get started. All searches are securely isolated to your credential.';

            const btnKey = document.createElement('button');
            btnKey.className = 'btn btn-primary';
            btnKey.textContent = 'Configure API Key';
            btnKey.addEventListener('click', () => openModal(elements.settingsModal));

            const btnPdf = document.createElement('button');
            btnPdf.className = 'btn btn-secondary';
            btnPdf.textContent = 'Upload PDF Document';
            btnPdf.addEventListener('click', () => openModal(elements.docsModal));

            actions.appendChild(btnKey);
            actions.appendChild(btnPdf);
        }

        welcome.appendChild(h2);
        welcome.appendChild(p);
        welcome.appendChild(actions);

        elements.chatMessages.appendChild(welcome);
    }

    function renderSessionHistory(messages) {
        elements.chatMessages.innerHTML = '';
        
        if (messages.length === 0) {
            const div = document.createElement('div');
            div.className = 'list-placeholder';
            div.textContent = 'Send a message to start this session.';
            elements.chatMessages.appendChild(div);
            return;
        }

        messages.forEach(msg => {
            appendMessageBubble(msg.role, msg.content, msg.sources);
        });
        
        scrollToBottom();
    }

    function appendMessageBubble(role, content, sources = []) {
        // Clean initial placeholders if exists
        const placeholders = elements.chatMessages.querySelectorAll('.list-placeholder, .welcome-container');
        placeholders.forEach(p => p.remove());

        const bubble = document.createElement('div');
        bubble.className = `message-bubble ${role === 'user' ? 'user' : 'assistant'}`;

        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';
        textDiv.textContent = content; // Security: always use textContent!

        bubble.appendChild(textDiv);

        // Append citations if available and role is assistant
        if (role !== 'user' && sources && sources.length > 0) {
            const sourcesContainer = document.createElement('div');
            sourcesContainer.className = 'sources-container';

            const title = document.createElement('div');
            title.className = 'sources-title';
            title.textContent = 'Citations & Sources';
            sourcesContainer.appendChild(title);

            const list = document.createElement('div');
            list.className = 'sources-list';

            sources.forEach(source => {
                const card = document.createElement('div');
                card.className = 'source-card';

                const header = document.createElement('div');
                header.className = 'source-header';

                const fileSpan = document.createElement('span');
                fileSpan.className = 'source-file';
                fileSpan.textContent = source.document_name || 'Unknown file';

                const metaSpan = document.createElement('span');
                metaSpan.className = 'source-meta';
                const page = source.page !== undefined ? source.page : 'N/A';
                const score = source.score !== undefined ? `L2 Dist: ${parseFloat(source.score).toFixed(3)}` : '';
                metaSpan.textContent = `Page ${page} ${score ? `| ${score}` : ''}`;

                header.appendChild(fileSpan);
                header.appendChild(metaSpan);
                card.appendChild(header);

                // Safe citation snippet
                if (source.snippet) {
                    const snippet = document.createElement('div');
                    snippet.className = 'source-snippet';
                    snippet.textContent = source.snippet.trim(); // Security: textContent
                    card.appendChild(snippet);
                }

                list.appendChild(card);
            });

            sourcesContainer.appendChild(list);
            bubble.appendChild(sourcesContainer);
        }

        elements.chatMessages.appendChild(bubble);
        scrollToBottom();
    }

    function scrollToBottom() {
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    }

    // --- Sending Messages & RAG Query Execution ---
    async function handleSendMessage(e) {
        if (e) e.preventDefault();

        const query = elements.promptInput.value.trim();
        if (!query) return;

        if (!state.apiKey) {
            showToast('API key is required to send messages.', 'error');
            return;
        }

        // Cancel existing pending query if any
        if (state.activeChatController) {
            state.activeChatController.abort();
        }

        // Clear input immediately and reset character counter
        elements.promptInput.value = '';
        elements.charCounter.textContent = '0 / 4000';
        elements.promptInput.style.height = 'auto';

        // Append user bubble
        appendMessageBubble('user', query);

        // Pre-generate a session UUID if we don't have one active
        if (!state.currentSessionId) {
            state.currentSessionId = uuidv4();
        }

        // Append processing indicator placeholder bubble
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble assistant loading-bubble';
        const loadText = document.createElement('div');
        loadText.className = 'message-text text-muted';
        loadText.textContent = 'Thinking...';
        bubble.appendChild(loadText);
        elements.chatMessages.appendChild(bubble);
        scrollToBottom();

        // Create AbortController for this request
        const controller = new AbortController();
        state.activeChatController = controller;

        const payload = {
            message: query,
            session_id: state.currentSessionId
        };

        try {
            const response = await apiRequest('/api/v1/chat/query', {
                method: 'POST',
                body: JSON.stringify(payload),
                signal: controller.signal
            });

            // Remove loading bubble
            bubble.remove();
            
            // Append real assistant response
            appendMessageBubble('assistant', response.answer, response.sources);
            
            // Reload sessions (message count & update times)
            state.sessionsPage = 1;
            await loadSessions();
        } catch (error) {
            if (error.name === 'AbortError') return;
            
            bubble.remove();
            let errMsg = error.message;
            if (error.status === 429) {
                errMsg = 'Rate limit exceeded. Please wait a moment before trying again.';
            }
            
            appendMessageBubble('assistant', `Failed to generate response: ${errMsg}`);
            showToast(errMsg, 'error');
        } finally {
            if (state.activeChatController === controller) {
                state.activeChatController = null;
            }
        }
    }

    // Basic client-side helper to make session IDs when starting fresh
    function uuidv4() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    // --- DOM Event Bindings ---
    function setupEventBindings() {
        // Modal toggles
        elements.btnOpenSettings.addEventListener('click', () => openModal(elements.settingsModal));
        elements.btnCloseSettings.addEventListener('click', () => closeModal(elements.settingsModal));
        
        elements.btnOpenDocs.addEventListener('click', () => {
            openModal(elements.docsModal);
            loadDocuments();
        });
        elements.btnCloseDocs.addEventListener('click', () => closeModal(elements.docsModal));

        // Welcome page link buttons
        elements.welcomeSetupKey.addEventListener('click', () => openModal(elements.settingsModal));
        elements.welcomeUploadPdf.addEventListener('click', () => openModal(elements.docsModal));

        // API Key actions
        elements.btnToggleKeyVisibility.addEventListener('click', () => {
            const type = elements.inputApiKey.type === 'password' ? 'text' : 'password';
            elements.inputApiKey.type = type;
            elements.btnToggleKeyVisibility.textContent = type === 'password' ? 'Show' : 'Hide';
        });

        elements.btnForgetKey.addEventListener('click', forgetApiKey);

        elements.btnTestConnection.addEventListener('click', async () => {
            const key = elements.inputApiKey.value.trim();
            if (!key) {
                showToast('Please enter a key to test.', 'error');
                return;
            }

            elements.btnTestConnection.disabled = true;
            elements.btnTestConnection.textContent = 'Testing...';
            
            try {
                // Call status to see if it allows the credentials
                const response = await fetch('/api/v1/vectorstore/status', {
                    headers: { 'X-API-Key': key }
                });
                
                if (response.status === 200) {
                    showToast('Connection verified successfully!', 'success');
                } else if (response.status === 401 || response.status === 403) {
                    showToast('Credentials rejected by server.', 'error');
                } else {
                    showToast(`Server returned status: ${response.status}`, 'error');
                }
            } catch (error) {
                showToast('Failed to contact server.', 'error');
            } finally {
                elements.btnTestConnection.disabled = false;
                elements.btnTestConnection.textContent = 'Test Connection';
            }
        });

        elements.settingsForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const key = elements.inputApiKey.value.trim();
            const strategy = document.querySelector('input[name="storage-type"]:checked').value;
            
            if (!key) {
                showToast('API key cannot be empty.', 'error');
                return;
            }

            saveApiKey(key, strategy);
            closeModal(elements.settingsModal);
            
            // Reload sidebar list
            state.sessionsPage = 1;
            loadSessions();
        });

        // New Chat trigger
        elements.btnNewChat.addEventListener('click', () => {
            if (!state.apiKey) {
                showToast('API key is required to create sessions.', 'error');
                openModal(elements.settingsModal);
                return;
            }
            state.currentSessionId = null;
            renderWelcomeScreen();
            
            // Clear sidebar highlighted items
            const items = elements.sessionsContainer.querySelectorAll('.session-item');
            items.forEach(item => item.classList.remove('active'));
        });

        // Chat Input / Composer setup
        elements.promptInput.addEventListener('input', () => {
            const val = elements.promptInput.value;
            elements.charCounter.textContent = `${val.length} / 4000`;
            
            // Auto grow input
            elements.promptInput.style.height = 'auto';
            elements.promptInput.style.height = `${elements.promptInput.scrollHeight}px`;
        });

        elements.promptInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
            }
        });

        elements.chatForm.addEventListener('submit', handleSendMessage);

        // Sidebar sessions pagination
        elements.btnPrevSessions.addEventListener('click', () => {
            if (state.sessionsPage > 1) {
                state.sessionsPage--;
                loadSessions();
            }
        });
        elements.btnNextSessions.addEventListener('click', () => {
            if ((state.sessionsPage * state.sessionsLimit) < state.sessionsTotal) {
                state.sessionsPage++;
                loadSessions();
            }
        });

        // Document Manager pagination
        elements.btnPrevDocs.addEventListener('click', () => {
            if (state.docsPage > 1) {
                state.docsPage--;
                loadDocuments();
            }
        });
        elements.btnNextDocs.addEventListener('click', () => {
            if (state.documents.length === state.docsLimit) {
                state.docsPage++;
                loadDocuments();
            }
        });

        // Drag & drop file upload
        const zone = elements.dragDropZone;
        
        zone.addEventListener('click', () => elements.fileInput.click());
        
        elements.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                uploadFile(e.target.files[0]);
            }
        });

        ['dragenter', 'dragover'].forEach(name => {
            zone.addEventListener(name, (e) => {
                e.preventDefault();
                zone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(name => {
            zone.addEventListener(name, (e) => {
                e.preventDefault();
                zone.classList.remove('dragover');
            }, false);
        });

        zone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            if (dt && dt.files.length > 0) {
                uploadFile(dt.files[0]);
            }
        });
    }

    // --- Setup app and run checker ---
    function init() {
        setupEventBindings();
        loadStoredApiKey();
        checkServerStatus();
        
        // Poll status indicator every 20 seconds
        setInterval(checkServerStatus, 20000);
        
        // Initial lists load if key found
        if (state.apiKey) {
            loadSessions();
        } else {
            renderWelcomeScreen();
            renderSessionsPlaceholder('No API key configured.');
        }
    }

    // Bootstrap
    document.addEventListener('DOMContentLoaded', init);

})();
