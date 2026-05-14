const form = document.querySelector("#run-form");
const directiveInput = document.querySelector("#directive");
const examplesInput = document.querySelector("#examples");
const targetCountInput = document.querySelector("#target-count");
const runButton = document.querySelector("#run-button");
const statusText = document.querySelector("#status-text");

const fields = {
  runMeta: document.querySelector("#run-meta"),
  runTitle: document.querySelector("#run-title"),
  selectedCount: document.querySelector("#selected-count"),
  rejectedCount: document.querySelector("#rejected-count"),
  sourceCount: document.querySelector("#source-count"),
  briefSummary: document.querySelector("#brief-summary"),
  briefTags: document.querySelector("#brief-tags"),
  sourcePlan: document.querySelector("#source-plan"),
  nextActions: document.querySelector("#next-actions"),
  imageGrid: document.querySelector("#image-grid"),
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await runWorkflow();
});

loadLatestRun();

async function loadLatestRun() {
  setStatus("Loading latest run");
  const response = await fetch("/api/runs/latest");
  const run = await response.json();
  renderRun(run);
  setStatus("Latest run loaded");
}

async function runWorkflow() {
  setStatus("Running workflow");
  runButton.disabled = true;
  const payload = {
    directive: directiveInput.value.trim(),
    examples: examplesInput.value
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean),
    target_count: Number(targetCountInput.value || 36),
  };

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new Error(`Run failed with ${response.status}`);
    }
    const run = await response.json();
    renderRun(run);
    setStatus("Workflow complete");
  } catch (error) {
    setStatus(error.message);
  } finally {
    runButton.disabled = false;
  }
}

function renderRun(run) {
  const created = new Date(run.created_at).toLocaleString();
  fields.runMeta.textContent = `${run.id} | ${created}`;
  fields.runTitle.textContent = run.request.directive;
  fields.selectedCount.textContent = run.selected_images.length;
  fields.rejectedCount.textContent = run.rejected_images.length;
  fields.sourceCount.textContent = run.source_plan.length;
  fields.briefSummary.textContent = run.style_brief.summary;

  fields.briefTags.innerHTML = "";
  [
    ...run.style_brief.keywords,
    ...run.style_brief.palette.slice(0, 2),
    ...run.style_brief.lighting.slice(0, 2),
  ].forEach((tag) => fields.briefTags.appendChild(tagNode(tag)));

  fields.sourcePlan.innerHTML = "";
  run.source_plan.forEach((plan) => {
    const item = document.createElement("div");
    item.className = "source-item";
    item.innerHTML = `
      <strong>${escapeHtml(plan.source)}</strong>
      <span>${escapeHtml(plan.goal)}</span>
    `;
    fields.sourcePlan.appendChild(item);
  });

  fields.nextActions.innerHTML = "";
  run.next_actions.forEach((action) => {
    const li = document.createElement("li");
    li.textContent = action;
    fields.nextActions.appendChild(li);
  });

  fields.imageGrid.innerHTML = "";
  run.selected_images.forEach((image) => fields.imageGrid.appendChild(imageCard(image)));
}

function imageCard(image) {
  const card = document.createElement("article");
  const score = image.score ? image.score.total.toFixed(3) : "0.000";
  card.className = "image-card";
  card.innerHTML = `
    <div class="thumb">
      <img src="${escapeAttribute(image.thumbnail_url)}" alt="">
      <span class="score-pill">${score}</span>
    </div>
    <div class="image-body">
      <h4>${escapeHtml(image.title)}</h4>
      <p>${escapeHtml(image.score ? image.score.rationale : image.notes)}</p>
      <div class="tag-list">${image.tags.slice(0, 4).map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("")}</div>
      <div class="card-footer">
        <span>${escapeHtml(image.source)}</span>
        <span>rights: ${escapeHtml(image.rights_risk)}</span>
      </div>
    </div>
  `;
  return card;
}

function tagNode(value) {
  const span = document.createElement("span");
  span.className = "tag";
  span.textContent = value;
  return span;
}

function setStatus(message) {
  statusText.textContent = message;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll("`", "&#096;");
}
