const byId = (id) => document.getElementById(id);

const state = {
  citations: [],
  selectedAtom: null,
};

function setVisible(id, visible) {
  byId(id).hidden = !visible;
}

function resetResults() {
  ["initialState", "loadingState", "answerRegion", "abstainedState", "errorState", "resultSummary"]
    .forEach((id) => setVisible(id, false));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function linkCitations(text) {
  return escapeHtml(text).replace(
    /\[(atom_[A-Za-z0-9_-]+)\]/g,
    '<button class="citation-link text-button" type="button" data-atom="$1">[$1]</button>'
  );
}

function selectCitation(atomId) {
  state.selectedAtom = atomId;
  document.querySelectorAll(".evidence-row").forEach((row) => {
    row.classList.toggle("selected", row.dataset.atom === atomId);
  });
  const selected = document.querySelector(`.evidence-row[data-atom="${CSS.escape(atomId)}"]`);
  selected?.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderEvidence(citations) {
  state.citations = citations;
  byId("evidenceCount").textContent = citations.length;
  setVisible("evidenceEmpty", citations.length === 0);
  byId("evidenceList").innerHTML = citations.map((citation, index) => `
    <article class="evidence-row" data-atom="${escapeHtml(citation.atom_id)}" tabindex="0">
      <div class="evidence-meta">
        <span>${index + 1}. Atom</span><code>${escapeHtml(citation.atom_id)}</code>
        <span>Feedback</span><code>${escapeHtml(citation.feedback_id || "Unavailable")}</code>
      </div>
      <p class="evidence-statement">${escapeHtml(citation.statement)}</p>
    </article>
  `).join("");
}

function renderAnswer(payload) {
  resetResults();
  setVisible("resultSummary", true);
  byId("confidence").textContent = Number(payload.retrieval.top_score).toFixed(3);
  byId("threshold").textContent = Number(payload.retrieval.abstain_threshold).toFixed(3);
  byId("citationCount").textContent = payload.citations.length;
  byId("resultStatus").textContent = payload.status === "answered" ? "Answered" : "Abstained";
  byId("resultSummary").classList.toggle("is-abstained", payload.status !== "answered");

  if (payload.status !== "answered") {
    setVisible("abstainedState", true);
    byId("abstainedCopy").textContent = payload.answer;
    renderEvidence([]);
    return;
  }

  setVisible("answerRegion", true);
  byId("answerCopy").innerHTML = linkCitations(payload.answer);
  byId("uncertaintyCopy").textContent = payload.uncertainty;
  const recommendations = payload.recommendations || [];
  setVisible("recommendationsRegion", recommendations.length > 0);
  byId("recommendationsList").innerHTML = recommendations.map((item) => `<li>${linkCitations(item)}</li>`).join("");
  renderEvidence(payload.citations || []);
}

function requestBody() {
  return {
    query: byId("query").value.trim(),
    top_k: Number(byId("topK").value),
    customer_segment: byId("customerSegment").value || null,
    product_area: byId("productArea").value || null,
    severity: byId("severity").value || null,
    source_type: byId("sourceType").value || null,
  };
}

async function submitQuery(event) {
  event?.preventDefault();
  const body = requestBody();
  if (body.query.length < 3) return;
  resetResults();
  setVisible("loadingState", true);
  renderEvidence([]);
  byId("submitButton").disabled = true;
  byId("submitButton").textContent = "Searching";

  try {
    const response = await fetch("/v1/answers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail?.message || payload.detail || "The answer service returned an error.");
    renderAnswer(payload);
  } catch (error) {
    resetResults();
    setVisible("errorState", true);
    byId("errorCopy").textContent = error.message;
  } finally {
    byId("submitButton").disabled = false;
    byId("submitButton").textContent = "Search";
  }
}

async function checkReadiness() {
  const node = byId("readyState");
  try {
    const response = await fetch("/ready");
    node.className = response.ok ? "ready is-ready" : "ready not-ready";
    node.lastElementChild.textContent = response.ok ? "Ready" : "Not ready";
  } catch {
    node.className = "ready not-ready";
    node.lastElementChild.textContent = "Offline";
  }
}

byId("queryForm").addEventListener("submit", submitQuery);
byId("query").addEventListener("input", (event) => {
  byId("characterCount").textContent = `${event.target.value.length} / 2000`;
});
byId("query").addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.key === "Enter") submitQuery(event);
});
byId("resetFilters").addEventListener("click", () => {
  ["customerSegment", "productArea", "severity", "sourceType"].forEach((id) => { byId(id).value = ""; });
  byId("topK").value = 8;
});
document.addEventListener("click", (event) => {
  const link = event.target.closest("[data-atom]");
  if (link) selectCitation(link.dataset.atom);
});
document.addEventListener("keydown", (event) => {
  const row = event.target.closest?.(".evidence-row");
  if (row && (event.key === "Enter" || event.key === " ")) selectCitation(row.dataset.atom);
});

byId("characterCount").textContent = `${byId("query").value.length} / 2000`;
checkReadiness();
