/**
 * Laser RAG QA — Frontend Application
 *
 * Connects to the /api/v1/chat SSE streaming endpoint via fetch,
 * renders streaming answers in real-time, displays collapsible
 * citation references, and provides retry/error handling.
 */
(function () {
    "use strict";

    // =========================================================================
    //  Configuration
    // =========================================================================
    const CHAT_URL = "/api/v1/chat";
    const RETRY_DELAY_MS = 2000;
    const MAX_RETRIES = 2;

    // =========================================================================
    //  DOM references
    // =========================================================================
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const chatArea = $("#chat-area");
    const welcome = $("#welcome");
    const questionInput = $("#question-input");
    const sendBtn = $("#send-btn");
    const sendIcon = $("#send-icon");
    const stopIcon = $("#stop-icon");
    const newChatBtn = $("#new-chat-btn");
    const statusIndicator = $("#status-indicator");
    const statusText = $("#status-text");

    // templates
    const tplUserMsg = $("#tpl-user-msg");
    const tplAssistantMsg = $("#tpl-assistant-msg");

    // =========================================================================
    //  State
    // =========================================================================
    let isStreaming = false;
    let abortController = null;
    // conversation history — accumulates across turns for context
    let conversationHistory = [];

    // =========================================================================
    //  Status indicator
    // =========================================================================
    function setStatus(state) {
        statusIndicator.className = "status-dot";
        switch (state) {
            case "active":
                statusIndicator.classList.add("status-active");
                statusText.textContent = "生成中";
                break;
            case "success":
                statusIndicator.classList.add("status-success");
                statusText.textContent = "就绪";
                break;
            case "error":
                statusIndicator.classList.add("status-error");
                statusText.textContent = "异常";
                break;
            default:
                statusIndicator.classList.add("status-idle");
                statusText.textContent = "就绪";
        }
    }

    // =========================================================================
    //  Streaming abort
    // =========================================================================
    function abortStream() {
        if (abortController) {
            abortController.abort();
            abortController = null;
        }
    }

    // =========================================================================
    //  Set streaming state — toggles send/stop button
    // =========================================================================
    function setStreamingState(active) {
        isStreaming = active;
        if (active) {
            sendIcon.style.display = "none";
            stopIcon.style.display = "block";
            sendBtn.style.background = "var(--error)";
            sendBtn.title = "停止生成 (Esc)";
        } else {
            sendIcon.style.display = "block";
            stopIcon.style.display = "none";
            sendBtn.style.background = "";
            sendBtn.title = "发送 (Enter)";
        }
    }

    // =========================================================================
    //  SSE parser: converts a fetch ReadableStream into SSE event objects
    // =========================================================================
    async function* parseSSEStream(response) {
        if (!response.ok) {
            const text = await response.text().catch(() => "");
            let detail = text;
            try {
                const j = JSON.parse(text);
                detail = j.detail || j.message || text;
            } catch (_) { /* not JSON */ }
            throw new Error(`HTTP ${response.status}: ${detail}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            // Keep the last potentially incomplete line in the buffer
            buffer = lines.pop() || "";

            let eventType = "message";
            let dataLines = [];

            for (const line of lines) {
                if (line.startsWith("event:")) {
                    eventType = line.slice(6).trim();
                } else if (line.startsWith("data:")) {
                    dataLines.push(line.slice(5).trim());
                } else if (line === "") {
                    // blank line = end of event
                    if (dataLines.length > 0) {
                        const raw = dataLines.join("\n");
                        let parsed;
                        try {
                            parsed = JSON.parse(raw);
                        } catch (_) {
                            parsed = { _raw: raw };
                        }
                        yield { event: eventType, data: parsed };
                    }
                    eventType = "message";
                    dataLines = [];
                }
            }
        }

        // flush remaining
        if (dataLines.length > 0) {
            const raw = dataLines.join("\n");
            let parsed;
            try {
                parsed = JSON.parse(raw);
            } catch (_) {
                parsed = { _raw: raw };
            }
            yield { event: eventType, data: parsed };
        }
    }

    // =========================================================================
    //  Message rendering
    // =========================================================================

    /** Create a user message DOM element. */
    function createUserMsg(text) {
        const clone = tplUserMsg.content.cloneNode(true);
        clone.querySelector(".msg-bubble-user").textContent = text;
        return clone.firstElementChild;
    }

    /** Create an assistant message DOM element (initially empty, streaming). */
    function createAssistantMsg() {
        const clone = tplAssistantMsg.content.cloneNode(true);
        const msg = clone.firstElementChild;
        // mark as streaming for cursor
        msg.querySelector(".msg-content").classList.add("streaming");
        return msg;
    }

    /** Append a loading indicator to the given message's content area. */
    function showLoading(msgEl) {
        const content = msgEl.querySelector(".msg-content");
        content.innerHTML =
            '<span class="msg-loading">' +
            '  <span class="msg-loading-dots">' +
            '    <span></span><span></span><span></span>' +
            '  </span>' +
            '  思考中...' +
            '</span>';
    }

    /** Render a simple Markdown-like text into the content element. */
    function renderMarkdown(el, text) {
        // Escape HTML first, then apply simple formatting
        let html = escapeHtml(text);

        // Bold: **text**
        html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

        // Inline code: `code`
        html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

        // Numbered headers: ### 标题
        html = html.replace(/^### (.+)$/gm, "<strong>$1</strong>");

        // Bullet points: - item  or  * item
        html = html.replace(/^[-*] (.+)$/gm, "• $1");

        // Numbered lists: preserve them with proper indentation
        html = html.replace(/^\d+\.\s(.+)$/gm, (_, content) => {
            return '<span style="display:block;padding-left:1em;">• ' + content + "</span>";
        });

        // Blank line → paragraph break
        html = html.replace(/\n\n+/g, "</p><p>");
        html = "<p>" + html + "</p>";
        // Remove empty paragraphs
        html = html.replace(/<p>\s*<\/p>/g, "");

        el.innerHTML = html;
    }

    /** Update the assistant message content (partial or full). */
    function updateMessageContent(msgEl, text) {
        const content = msgEl.querySelector(".msg-content");
        renderMarkdown(content, text);
        scrollToBottom();
    }

    /** Mark the assistant message as complete (remove cursor, show finish info). */
    function finalizeMessage(msgEl, finishReason) {
        const content = msgEl.querySelector(".msg-content");
        content.classList.remove("streaming");

        const meta = msgEl.querySelector(".msg-finish-reason");
        const labels = { stop: "✔ 完成", length: "⚠ 截断", error: "✘ 错误" };
        meta.textContent = labels[finishReason] || finishReason || "";
    }

    /** Attach references to an assistant message. */
    function attachReferences(msgEl, references) {
        if (!references || references.length === 0) return;

        const refsContainer = msgEl.querySelector(".msg-refs");
        const refsList = refsContainer.querySelector(".refs-list");
        const refsCount = refsContainer.querySelector(".refs-count");
        const toggleBtn = refsContainer.querySelector(".refs-toggle");

        // count KB vs web refs
        const kbCount = references.filter((r) => !r.is_web).length;
        const webCount = references.filter((r) => r.is_web).length;
        const parts = [];
        if (kbCount > 0) parts.push(`📚 ${kbCount} 条`);
        if (webCount > 0) parts.push(`🌐 ${webCount} 条`);
        refsCount.textContent = parts.join(" + ");

        // build reference cards
        refsList.innerHTML = "";
        for (const ref of references) {
            const card = document.createElement("div");
            card.className = "ref-card";

            // source badge
            const sourceTag = ref.is_web
                ? '<span class="ref-card-tag ref-card-tag-web">🌐 网络来源</span>'
                : '<span class="ref-card-tag ref-card-tag-kb">📚 知识库</span>';

            // URL link for web refs
            const urlBlock = ref.url
                ? '<a class="ref-card-url" href="' + escapeHtml(ref.url) +
                  '" target="_blank" rel="noopener noreferrer">' +
                  escapeHtml(ref.url) + "</a>"
                : "";

            card.innerHTML =
                '<div class="ref-card-header">' +
                '  <span class="ref-card-index">' + ref.index + "</span>" +
                '  <span class="ref-card-source">' + escapeHtml(ref.source) + "</span>" +
                sourceTag +
                (ref.page
                    ? '  <span class="ref-card-page">第' + ref.page + "页</span>"
                    : "") +
                "</div>" +
                '<div class="ref-card-text">' + escapeHtml(ref.text) + "</div>" +
                urlBlock;
            refsList.appendChild(card);
        }

        // toggle behaviour
        refsContainer.hidden = false;
        toggleBtn.addEventListener("click", () => {
            const hidden = refsList.hidden;
            refsList.hidden = !hidden;
            toggleBtn.classList.toggle("open", hidden);
            toggleBtn.querySelector(".refs-toggle-label").textContent = hidden
                ? "收起来源"
                : "参考来源";
            scrollToBottom();
        });
    }

    /** Show error info inside an assistant message. */
    function showMessageError(msgEl, errorText, partialContent) {
        const content = msgEl.querySelector(".msg-content");
        content.classList.remove("streaming");

        if (partialContent) {
            // keep the partial content visible
        } else if (!content.textContent.trim()) {
            content.textContent = "（未能生成回答）";
        }

        // add error block
        const errEl = document.createElement("div");
        errEl.className = "msg-error";
        errEl.textContent = "⚠ " + errorText;
        content.appendChild(errEl);

        const btnRetry = msgEl.querySelector(".btn-retry");
        btnRetry.hidden = false;
    }

    // =========================================================================
    //  Scroll helper
    // =========================================================================
    function scrollToBottom() {
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    // =========================================================================
    //  Escape HTML
    // =========================================================================
    function escapeHtml(str) {
        const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
        return String(str).replace(/[&<>"']/g, (c) => map[c]);
    }

    // =========================================================================
    //  Core: send question and consume SSE stream
    // =========================================================================
    async function sendQuestion(question) {
        if (!question.trim() || isStreaming) return;

        // abort any existing stream
        abortStream();

        setStreamingState(true);
        setStatus("active");
        questionInput.disabled = true;

        // hide welcome
        if (welcome) welcome.hidden = true;

        // clear input
        const questionText = question.trim();
        questionInput.value = "";
        questionInput.style.height = "auto";  // reset textarea height

        // append user message
        const userMsg = createUserMsg(questionText);
        chatArea.appendChild(userMsg);
        scrollToBottom();

        // append assistant message placeholder
        const assistantMsg = createAssistantMsg();
        chatArea.appendChild(assistantMsg);
        showLoading(assistantMsg);
        scrollToBottom();

        // prepare request
        abortController = new AbortController();
        let fullText = "";
        let references = null;
        let finishReason = "stop";
        let streamCompleted = false;  // set to true when "done" SSE event received
        let retries = 0;

        while (retries <= MAX_RETRIES) {
            try {
                const payload = {
                    question: questionText,
                    history: conversationHistory.length > 0 ? conversationHistory : undefined,
                    enable_web_search: document.getElementById("web-search-checkbox").checked,
                };

                const response = await fetch(CHAT_URL, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                    signal: abortController.signal,
                });

                let firstToken = true;
                streamCompleted = false;

                for await (const sse of parseSSEStream(response)) {
                    if (firstToken) {
                        // clear loading indicator on first content
                        assistantMsg.querySelector(".msg-content").innerHTML = "";
                        assistantMsg.querySelector(".msg-content").classList.add("streaming");
                        firstToken = false;
                    }

                    switch (sse.event) {
                        case "token":
                            fullText += sse.data.content || "";
                            updateMessageContent(assistantMsg, fullText);
                            break;

                        case "references":
                            references = sse.data.references || [];
                            if (references.length > 0) {
                                attachReferences(assistantMsg, references);
                            }
                            break;

                        case "done":
                            finishReason = sse.data.finish_reason || "stop";
                            streamCompleted = true;
                            // Immediately stop consuming the stream — answer is complete
                            break;

                        case "error":
                            throw new Error(
                                sse.data.message || "服务器返回错误"
                            );
                    }

                    if (streamCompleted) break;  // exit for-await immediately
                }

                // stream completed successfully
                break;

            } catch (err) {
                if (err.name === "AbortError") {
                    // user aborted — keep partial content if any, don't retry
                    if (!fullText) fullText = "";
                    finishReason = fullText ? "stop" : "error";
                    break;
                }

                // If we already have answer content, don't retry — treat as success
                if (fullText && fullText.trim().length > 0) {
                    finishReason = "stop";
                    break;
                }

                retries++;
                if (retries > MAX_RETRIES) {
                    finishReason = "error";
                    showMessageError(
                        assistantMsg,
                        err.message || "连接失败，请稍后重试",
                        fullText
                    );
                } else {
                    // show retry status in meta
                    const meta = assistantMsg.querySelector(".msg-finish-reason");
                    meta.textContent = `重试中 (${retries}/${MAX_RETRIES})...`;
                    await sleep(RETRY_DELAY_MS);
                }
            }
        }

        // finalize
        finalizeMessage(assistantMsg, finishReason);

        // update conversation history (keep last 10 turns to limit context)
        if (finishReason !== "error" && fullText) {
            conversationHistory.push({ role: "user", content: questionText });
            conversationHistory.push({ role: "assistant", content: fullText });
            // prune to last 10 turns (20 messages)
            if (conversationHistory.length > 20) {
                conversationHistory = conversationHistory.slice(-20);
            }
        }

        // show retry button on error
        if (finishReason === "error") {
            const btnRetry = assistantMsg.querySelector(".btn-retry");
            btnRetry.hidden = false;
            btnRetry.addEventListener("click", () => {
                // remove the failed assistant message and re-send
                assistantMsg.remove();
                sendQuestion(questionText);
            });
        }

        // clean up
        abortController = null;
        setStreamingState(false);
        setStatus(finishReason === "error" ? "error" : "success");
        questionInput.disabled = false;
        questionInput.focus();
    }

    // =========================================================================
    //  Utility: async sleep
    // =========================================================================
    function sleep(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }

    // =========================================================================
    //  Event listeners
    // =========================================================================

    // Send / Stop button
    sendBtn.addEventListener("click", () => {
        if (isStreaming) {
            // Stop generation
            abortStream();
            setStreamingState(false);
            setStatus("success");
            questionInput.disabled = false;
            questionInput.focus();
        } else {
            sendQuestion(questionInput.value);
        }
    });

    // Keyboard: Enter to send, Shift+Enter for newline, Esc to stop
    questionInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (isStreaming) {
                abortStream();
                setStreamingState(false);
                setStatus("success");
                questionInput.disabled = false;
                questionInput.focus();
            } else {
                sendQuestion(questionInput.value);
            }
        }
        if (e.key === "Escape" && isStreaming) {
            e.preventDefault();
            abortStream();
            setStreamingState(false);
            setStatus("success");
            questionInput.disabled = false;
            questionInput.focus();
        }
    });

    // Auto-resize textarea
    questionInput.addEventListener("input", () => {
        questionInput.style.height = "auto";
        questionInput.style.height =
            Math.min(questionInput.scrollHeight, 150) + "px";
    });

    // Quick prompt chips
    chatArea.addEventListener("click", (e) => {
        const chip = e.target.closest(".prompt-chip");
        if (chip) {
            const prompt = chip.dataset.prompt;
            if (prompt) {
                questionInput.value = prompt;
                sendQuestion(prompt);
            }
        }
    });

    // New conversation button
    newChatBtn.addEventListener("click", () => {
        // Abort any ongoing stream
        abortStream();
        setStreamingState(false);
        // Clear conversation history
        conversationHistory = [];
        // Clear chat messages (keep only welcome)
        const messages = chatArea.querySelectorAll(".message");
        messages.forEach((m) => m.remove());
        // Show welcome
        if (welcome) welcome.hidden = false;
        // Reset UI state
        setStatus("idle");
        questionInput.value = "";
        questionInput.style.height = "auto";
        questionInput.disabled = false;
        questionInput.focus();
    });

    // =========================================================================
    //  Initialisation
    // =========================================================================
    questionInput.focus();
    setStatus("idle");
})();
