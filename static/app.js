const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const videoFeed = document.getElementById("videoFeed");
const statusGrid = document.getElementById("statusGrid");
const activeList = document.getElementById("activeList");
const recentList = document.getElementById("recentList");
const metricsBox = document.getElementById("metricsBox");
const runPill = document.getElementById("runPill");
const logDate = document.getElementById("logDate");
const logTag = document.getElementById("logTag");
const logSearch = document.getElementById("logSearch");
const searchLogsBtn = document.getElementById("searchLogsBtn");
const logsList = document.getElementById("logsList");
let videoStreamActive = false;

async function postJSON(path) {
  const response = await fetch(path, { method: "POST" });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function sevClass(sev) {
  return `sev-${sev}`;
}

function renderStatus(status) {
  const entries = [
    ["Running", String(status.running)],
    ["Source", status.source],
    ["FPS", String(status.fps)],
    ["Active incidents", String(status.active_incidents)],
    ["Total incidents", String(status.total_incidents)],
    ["Cameras online", String(status.cameras_online)],
    ["People in frame", String(status.people_in_frame)],
  ];

  statusGrid.innerHTML = entries
    .map(
      ([k, v]) => `
      <div class="kv">
        <div class="k">${k}</div>
        <div class="v">${v}</div>
      </div>
    `,
    )
    .join("");

  runPill.className = `pill ${status.running ? "online" : "offline"}`;
  runPill.textContent = status.running ? "ONLINE" : "OFFLINE";
}

function renderIncidents(listEl, incidents) {
  if (!incidents.length) {
    listEl.innerHTML = "<li>No incidents detected</li>";
    return;
  }

  listEl.innerHTML = incidents
    .map(
      (item) => `
      <li class="${sevClass(item.severity)}">
        <strong>${item.disaster_type}</strong>
        <div>${item.message}</div>
        <small>${new Date(item.created_at).toLocaleString()} | ${item.location_tag}</small>
      </li>
    `,
    )
    .join("");
}

function renderLogs(logs) {
  if (!logs.length) {
    logsList.innerHTML = "<li>No logs found for selected filters</li>";
    return;
  }

  logsList.innerHTML = logs
    .map(
      (item) => `
      <li class="${sevClass(item.severity)}">
        <strong>${item.disaster_type}</strong>
        <div>${item.message}</div>
        <small>${new Date(item.created_at).toLocaleString()} | ${item.location_tag} | tags: ${item.tags.join(", ")}</small>
      </li>
    `,
    )
    .join("");
}

function syncVideoFeed(running) {
  if (running) {
    if (!videoStreamActive) {
      videoFeed.src = "/api/video_feed";
      videoStreamActive = true;
    }
    return;
  }

  if (videoStreamActive) {
    videoFeed.removeAttribute("src");
    videoStreamActive = false;
  }
}

async function refreshLogs() {
  const params = new URLSearchParams();
  if (logDate.value) params.set("date", logDate.value);
  if (logTag.value) params.set("tag", logTag.value);
  if (logSearch.value.trim()) params.set("search", logSearch.value.trim());
  params.set("limit", "150");

  try {
    const response = await fetch(`/api/logs?${params.toString()}`);
    if (!response.ok) return;

    const data = await response.json();
    renderLogs(data.logs || []);
  } catch (err) {
    console.error(err);
  }
}

async function refreshDashboard() {
  try {
    const response = await fetch("/api/dashboard");
    if (!response.ok) return;

    const data = await response.json();
    renderStatus(data.status);
    renderIncidents(activeList, data.active_incidents);
    renderIncidents(recentList, data.recent_incidents);
    metricsBox.textContent = JSON.stringify(data.metrics, null, 2);
    syncVideoFeed(data.status.running);
  } catch (err) {
    console.error(err);
  }
}

startBtn.addEventListener("click", async () => {
  try {
    await postJSON("/api/start");
    await refreshDashboard();
  } catch (err) {
    console.error(err);
  }
});

stopBtn.addEventListener("click", async () => {
  try {
    await postJSON("/api/stop");
    await refreshDashboard();
  } catch (err) {
    console.error(err);
  }
});

searchLogsBtn.addEventListener("click", async () => {
  await refreshLogs();
});

logDate.value = new Date().toISOString().slice(0, 10);

refreshDashboard();
refreshLogs();
setInterval(refreshDashboard, 2500);
setInterval(refreshLogs, 15000);
