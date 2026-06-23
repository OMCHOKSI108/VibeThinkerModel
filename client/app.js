"use strict";

/**
 * VibeThinker Local Chat — frontend logic.
 * Talks to the FastAPI backend via:
 *   GET  /health              -> model status badge
 *   POST /api/chat/stream     -> SSE-formatted stream, parsed manually (fetch, not EventSource,
 *                                 because EventSource can't send a POST body)
 *   POST /graphql             -> resetSession mutation on Clear
 */

const chatArea = document.getElementById("chat-area");
const emptyState = document.getElementById("empty-state");
const messageInput = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const stopBtn = document.getElementById("stop-btn");
const clearBtn = document.getElementById("clear-btn");
const modelBadge = document.getElementById("model-badge");
const settingsToggle = document.getElementById("settings-toggle");
const settingsClose = document.getElementById("settings-close");
const settingsDrawer = document.getElementById("settings-drawer");
const settingMaxTokens = document.getElementById("setting-max-tokens");
const settingTemperature = document.getElementById("setting-temperature");
const settingTopP = document.getElementById("setting-top-p");
const settingSystemPrompt = document.getElementById("setting-system-prompt");

const sessionId = crypto.randomUUID();
let activeAbortController = null;

// --- Model status badge -----------------------------------------------------

async function refreshModelBadge() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    if (data.model_loaded) {
      modelBadge.textContent = `ready · ${data.device}`;
      modelBadge.className = "badge badge-ready";
    } else {
      modelBadge.textContent = data.load_error ? "model failed to load" : "model loading...";
      modelBadge.className = "badge badge-error";
    }
  } catch (err) {
    modelBadge.textContent = "backend unreachable";
    modelBadge.className = "badge badge-error";
  }
}

refreshModelBadge();
setInterval(refreshModelBadge, 10000);

// --- Settings drawer ---------------------------------------------------------

settingsToggle.addEventListener("click", () => settingsDrawer.classList.remove("hidden"));
settingsClose.addEventListener("click", () => settingsDrawer.classList.add("hidden"));

// --- Markdown rendering -------------------------------------------------------

function renderMarkdown(rawText) {
  const html = window.marked.parse(rawText, { breaks: true });
  return html;
}

function attachCopyButtons(bubbleEl) {
  bubbleEl.querySelectorAll("pre").forEach((pre) => {
    if (pre.parentElement.classList.contains("code-block-wrapper")) return;
    const wrapper = document.createElement("div");
    wrapper.className = "code-block-wrapper";
    pre.replaceWith(wrapper);
    wrapper.appendChild(pre);

    const copyBtn = document.createElement("button");
    copyBtn.className = "copy-btn";
    copyBtn.textContent = "Copy";
    copyBtn.addEventListener("click", async () => {
      const codeEl = pre.querySelector("code");
      const text = codeEl ? codeEl.textContent : pre.textContent;
      await navigator.clipboard.writeText(text);
      copyBtn.textContent = "Copied";
      copyBtn.classList.add("copied");
      setTimeout(() => {
        copyBtn.textContent = "Copy";
        copyBtn.classList.remove("copied");
      }, 1500);
    });
    wrapper.appendChild(copyBtn);
  });
}

// --- Chat message rendering ---------------------------------------------------

function createMessageEl(role) {
  emptyState.classList.add("hidden");
  const messageEl = document.createElement("div");
  messageEl.className = `message ${role}`;

  const roleEl = document.createElement("div");
  roleEl.className = "message-role";
  roleEl.textContent = role === "user" ? "You" : "VibeThinker";
  messageEl.appendChild(roleEl);

  const bubbleEl = document.createElement("div");
  bubbleEl.className = "message-bubble";
  messageEl.appendChild(bubbleEl);

  chatArea.appendChild(messageEl);
  scrollToBottom();
  return { messageEl, bubbleEl };
}

function createThinkingBar() {
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant";

  const roleEl = document.createElement("div");
  roleEl.className = "message-role";
  roleEl.textContent = "VibeThinker";
  wrapper.appendChild(roleEl);

  const bar = document.createElement("div");
  bar.className = "thinking-bar";
  bar.innerHTML = `
    <div class="thinking-dots"><span></span><span></span><span></span></div>
    <span class="thinking-label">Thinking...</span>
    <div class="progress-track"><div class="progress-fill"></div></div>
  `;
  wrapper.appendChild(bar);
  chatArea.appendChild(wrapper);
  scrollToBottom();
  return { wrapper, label: bar.querySelector(".thinking-label") };
}

function scrollToBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

// --- SSE stream parsing --------------------------------------------------------

/**
 * Manually parses an SSE byte stream from a fetch() Response body.
 * Each event is a block separated by a blank line, with `event:` and `data:` lines.
 */
async function* parseSseStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);

      let eventName = "message";
      let dataLine = "";
      for (const line of rawEvent.split("\n")) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLine += line.slice(5).trim();
      }
      if (!dataLine) continue;
      try {
        yield { event: eventName, data: JSON.parse(dataLine) };
      } catch {
        // Ignore malformed frames rather than breaking the whole stream.
      }
    }
  }
}

// --- Send flow -------------------------------------------------------------------

function setGenerating(isGenerating) {
  sendBtn.disabled = isGenerating;
  stopBtn.classList.toggle("hidden", !isGenerating);
}

async function sendMessage() {
  const text = messageInput.value.trim();
  if (!text) return;

  messageInput.value = "";
  messageInput.style.height = "auto";

  createMessageEl("user").bubbleEl.textContent = text;

  const thinking = createThinkingBar();
  setGenerating(true);

  activeAbortController = new AbortController();
  let assistantBubble = null;
  let assistantText = "";

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: activeAbortController.signal,
      body: JSON.stringify({
        session_id: sessionId,
        message: text,
        system_prompt: settingSystemPrompt.value.trim(),
        max_new_tokens: parseInt(settingMaxTokens.value, 10),
        temperature: parseFloat(settingTemperature.value),
        top_p: parseFloat(settingTopP.value),
      }),
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || body.error || `Request failed (${response.status})`);
    }

    for await (const { event, data } of parseSseStream(response)) {
      if (event === "status") {
        thinking.label.textContent = data.message || data.stage;
      } else if (event === "token") {
        if (!assistantBubble) {
          thinking.wrapper.remove();
          assistantBubble = createMessageEl("assistant").bubbleEl;
        }
        assistantText += data.text;
        assistantBubble.innerHTML = renderMarkdown(assistantText);
        attachCopyButtons(assistantBubble);
        scrollToBottom();
      } else if (event === "error") {
        thinking.wrapper.remove();
        const bubble = createMessageEl("assistant").bubbleEl;
        bubble.classList.add("error-bubble");
        bubble.textContent = data.message || "Generation failed.";
      } else if (event === "done") {
        thinking.wrapper.remove();
        if (!assistantBubble) {
          // Model produced no visible output (e.g. only a think block).
          createMessageEl("assistant").bubbleEl.textContent = "(no response)";
        }
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      thinking.wrapper.remove();
      const bubble = createMessageEl("assistant").bubbleEl;
      bubble.classList.add("error-bubble");
      bubble.textContent = err.message || "Connection error.";
    } else if (!assistantBubble) {
      thinking.wrapper.remove();
    }
  } finally {
    setGenerating(false);
    activeAbortController = null;
  }
}

function stopGeneration() {
  if (activeAbortController) {
    activeAbortController.abort();
  }
}

function clearConversation() {
  chatArea.innerHTML = "";
  chatArea.appendChild(emptyState);
  emptyState.classList.remove("hidden");

  fetch("/graphql", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: `mutation($id: String!) { resetSession(sessionId: $id) { sessionId existed } }`,
      variables: { id: sessionId },
    }),
  }).catch(() => {});
}

// --- Event wiring -------------------------------------------------------------

sendBtn.addEventListener("click", sendMessage);
stopBtn.addEventListener("click", stopGeneration);
clearBtn.addEventListener("click", clearConversation);

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

messageInput.addEventListener("input", () => {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 200)}px`;
});
