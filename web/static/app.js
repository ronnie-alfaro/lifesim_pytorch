const ACTIONS = ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "EAT", "DRINK", "REST", "WAIT", "ATTACK", "GATHER", "MATE"];
const MOVEMENT_ACTIONS = new Set(["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"]);
const WORLD_CELL_SIZE = 20;

const ui = Object.fromEntries([
  "world", "connection-dot", "run-label", "tick-label", "status-label", "tick-progress",
  "play-button", "pause-button", "step-button", "cancel-button", "speed-input", "speed-label",
  "humans-alive", "animals-alive", "resources-count", "average-loss", "training-message",
  "completion-summary", "termination-label", "survivor-summary", "run-comparison", "learning-numbers", "learned-explanation",
  "next-run-button", "comparison-chart-wrap", "comparison-chart",
  "apply-config-button", "human-count-input", "human-count-label",
  "animal-count-input", "animal-count-label", "human-brain-input",
  "human-brain-label", "animal-brain-input", "animal-brain-label",
  "architecture-preview", "builder-help", "use-brb-input", "brb-description",
  "all-humans-button", "all-animals-button",
  "agent-name", "agent-kind", "agent-empty", "group-overview", "group-chart", "group-summary",
  "agent-details", "rpg-profile", "rpg-avatar", "rpg-title", "favorite-action", "agent-story", "rpg-facts", "vitals", "agent-action",
  "agent-reward", "agent-loss", "agent-mode", "agent-replay", "agent-updates", "epsilon-explanation",
  "brain-network", "brain-network-caption", "brain-view-label", "spatial-memory",
  "q-values", "observations", "reward-components", "event-stream", "toast"
].map(id => [id, document.getElementById(id)]));

let state = null;
let selectedAgentId = null;
let selectedScope = "human";
let pollTimer = null;
let configurationDirty = false;
const previousWeightMeans = new Map();

async function api(path, options = {}) {
  const response = await fetch(path, {cache: "no-store", ...options});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

async function poll() {
  try {
    state = await api("/api/state");
    ui["connection-dot"].classList.add("online");
    render();
    schedulePoll(state.status === "running" ? 160 : 450);
  } catch (error) {
    ui["connection-dot"].classList.remove("online");
    showError(`Conexión perdida: ${error.message}`);
    schedulePoll(1000);
  }
}

function schedulePoll(delay) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(poll, delay);
}

async function control(action, value = undefined) {
  try {
    state = await api("/api/control", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({action, value})
    });
    render();
    schedulePoll(80);
  } catch (error) {
    showError(error.message);
  }
}

function render() {
  if (!state) return;
  ui["run-label"].textContent = `Experimento ${pad(state.experiment_id)} · Run ${pad(state.run_number)}`;
  ui["tick-label"].textContent = `Tick ${state.tick.toLocaleString()} / ${state.num_ticks.toLocaleString()}`;
  ui["tick-progress"].style.width = `${(state.tick / state.num_ticks) * 100}%`;
  ui["status-label"].textContent = statusText(state.status);
  ui["status-label"].className = `status-badge ${state.status}`;
  ui["play-button"].disabled = state.status !== "paused";
  ui["pause-button"].disabled = state.status !== "running";
  ui["step-button"].disabled = state.status !== "paused";
  ui["cancel-button"].disabled = !state.can_cancel_experiment;
  ui["cancel-button"].textContent = state.status === "finalizing" ? "Cancelando…" : "Cancelar experimento";
  ui["next-run-button"].disabled = !state.can_start_next_run;
  ui["next-run-button"].textContent = state.status === "preparing" ? "Preparando mundo…" : "↻ Siguiente ciclo";
  ui["speed-input"].value = state.ticks_per_second;
  ui["speed-label"].textContent = `${format(state.ticks_per_second, 0)} ticks/s`;
  renderExperimentBuilder();

  const summary = state.summary;
  ui["humans-alive"].textContent = `${summary.living_humans} / ${state.agents.filter(agent => agent.type === "human").length}`;
  ui["animals-alive"].textContent = `${summary.living_animals} / ${state.agents.filter(agent => agent.type === "animal").length}`;
  ui["resources-count"].textContent = `${summary.food_remaining} / ${summary.water_remaining}`;
  ui["average-loss"].textContent = summary.average_loss == null ? "—" : format(summary.average_loss, 4);
  renderWorld();
  renderInspector();
  renderEvents();
  renderTrainingState();
  renderCompletionSummary();
  if (state.error) showError(state.error);
}

function renderExperimentBuilder() {
  const config = state.experiment_config;
  const brb = state.brb;
  if (!configurationDirty) {
    ui["human-count-input"].value = config.num_humans;
    ui["animal-count-input"].value = config.num_animals;
    ui["human-brain-input"].value = config.human_hidden_sizes.at(-1);
    ui["animal-brain-input"].value = config.animal_hidden_sizes.at(-1);
  }
  ui["use-brb-input"].disabled = !brb || !state.can_configure_experiment;
  if (!brb) ui["use-brb-input"].checked = false;
  applyBrbMinimums();
  updateBuilderLabels();
  ui["apply-config-button"].disabled = !state.can_configure_experiment;
  const completion = brb?.final_humans == null
    ? ""
    : ` · ${brb.final_humans}/${brb.initial_humans} humanos completaron el ciclo`;
  ui["brb-description"].textContent = brb
    ? `Campeón: experimento ${pad(brb.experiment_id)}, run ${pad(brb.run_number)}${completion} · supervivencia media ${format(brb.mean_human_survival, 1)}, mediana ${format(brb.median_human_survival, 1)}.${brb.learning_contract_current ? "" : ` Pesos base v${brb.source_reward_version}; reward v${brb.current_reward_version} reinicia Adam/replay y solo lo reemplazará un resultado compatible mejor.`}`
    : "Todavía no existe un run completo compatible para usar como campeón.";
  ui["builder-help"].textContent = state.can_configure_experiment
    ? (ui["use-brb-input"].checked
      ? "Copia los pesos del campeón. Mundo, cuerpos, Adam y replay Horde comienzan limpios para comparar el nuevo experimento."
      : "Crea un experimento independiente con brains nuevos. Más agentes y neuronas requieren más CPU.")
    : "Bloqueado durante el run. Termina el ciclo para crear otro experimento; Siguiente ciclo conserva los brains actuales.";
}

function applyBrbMinimums() {
  const brb = state?.brb;
  const enabled = Boolean(brb && ui["use-brb-input"].checked);
  const humanMinimum = enabled ? Number(brb.minimum_human_width) : 8;
  const animalMinimum = enabled ? Number(brb.minimum_animal_width) : 8;
  ui["human-brain-input"].min = humanMinimum;
  ui["animal-brain-input"].min = animalMinimum;
  if (Number(ui["human-brain-input"].value) < humanMinimum) ui["human-brain-input"].value = humanMinimum;
  if (Number(ui["animal-brain-input"].value) < animalMinimum) ui["animal-brain-input"].value = animalMinimum;
}

function updateBuilderLabels() {
  const humans = Number(ui["human-count-input"].value);
  const animals = Number(ui["animal-count-input"].value);
  const humanWidth = Number(ui["human-brain-input"].value);
  const animalWidth = Number(ui["animal-brain-input"].value);
  ui["human-count-label"].textContent = humans;
  ui["animal-count-label"].textContent = animals;
  ui["human-brain-label"].textContent = `${humanWidth} neuronas/capa`;
  ui["animal-brain-label"].textContent = `${animalWidth} neuronas`;
  ui["architecture-preview"].textContent = `Humano: supervivencia 15→${Math.max(8, humanWidth / 2)}, espacio social 29→${humanWidth}, fusión ${humanWidth}→11 · Animal: supervivencia 15, espacio social 27→${animalWidth} · ${humans + animals} brains v2`;
}

function renderWorld() {
  const {width, height, food, water, obstacles, stockpiles = []} = state.grid;
  const canvas = ui.world;
  const cellSize = WORLD_CELL_SIZE;
  canvas.width = width * cellSize;
  canvas.height = height * cellSize;
  canvas.style.aspectRatio = `${width} / ${height}`;
  const context = canvas.getContext("2d");
  context.imageSmoothingEnabled = false;

  // Base terrain is a matrix; stockpiles and agents are semantic overlays.
  const matrix = Array.from({length: height}, () => new Uint8Array(width));
  for (const [x, y] of obstacles) matrix[y][x] = 1;
  for (const [x, y] of food) matrix[y][x] = 2;
  for (const [x, y] of water) matrix[y][x] = 3;
  const paths = buildVillagePaths(width, height, stockpiles);

  context.fillStyle = "#ced8b7";
  context.fillRect(0, 0, canvas.width, canvas.height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      drawMatrixCell(context, matrix[y][x], x, y, cellSize, matrix, paths.has(`${x},${y}`), state.tick);
    }
  }
  for (const stockpile of stockpiles) drawStockpile(context, stockpile, cellSize, state.tick);
  const detailed = state.agents.length <= 600;
  const agents = [...state.agents].sort((first, second) => {
    const layer = agent => agent.id === selectedAgentId ? 4 : !agent.alive ? 0 : agent.dependent_ticks_remaining > 0 ? 3 : 2;
    return layer(first) - layer(second) || first.id.localeCompare(second.id);
  });
  for (const agent of agents) {
    drawPixelAgent(context, agent, cellSize, agent.id === selectedAgentId, state.tick, detailed);
  }
}

function buildVillagePaths(width, height, stockpiles) {
  const paths = new Set();
  const add = (x, y) => {
    if (x >= 0 && y >= 0 && x < width && y < height) paths.add(`${x},${y}`);
  };
  for (const stockpile of stockpiles) {
    const radius = 2 + Math.min(4, Math.floor(Number(stockpile.food || 0) / 2));
    for (let step = -radius; step <= radius; step += 1) {
      add(stockpile.x + step, stockpile.y);
      add(stockpile.x, stockpile.y + step);
    }
    for (let dx = -1; dx <= 1; dx += 1) {
      for (let dy = -1; dy <= 1; dy += 1) add(stockpile.x + dx, stockpile.y + dy);
    }
  }
  if (stockpiles.length > 1) {
    const [first, second] = stockpiles;
    const left = Math.min(first.x, second.x);
    const right = Math.max(first.x, second.x);
    const top = Math.min(first.y, second.y);
    const bottom = Math.max(first.y, second.y);
    for (let x = left; x <= right; x += 1) add(x, first.y);
    for (let y = top; y <= bottom; y += 1) add(second.x, y);
  }
  return paths;
}

function drawStockpile(context, stockpile, size, tick) {
  const px = stockpile.x * size;
  const py = stockpile.y * size;
  const food = Number(stockpile.food || 0);
  const level = food === 0 ? 0 : food < 3 ? 1 : food < 6 ? 2 : 3;
  context.fillStyle = "rgba(56,42,29,.24)";
  context.fillRect(px + 1, py + 16, 18, 3);
  context.fillStyle = "#6f4c2d";
  context.fillRect(px + 2, py + 9, 14, 8);
  context.fillStyle = "#b9854e";
  context.fillRect(px + 3, py + 10, 12, 2);
  context.fillRect(px + 5, py + 14, 9, 1);
  context.fillStyle = "#432f22";
  context.fillRect(px + 8, py + 9, 2, 8);
  if (level >= 1) {
    context.fillStyle = "#d8b66a";
    context.fillRect(px + 1, py + 12, 4, 5);
    context.fillStyle = "#8e6b3f";
    context.fillRect(px + 2, py + 11, 2, 1);
  }
  if (level >= 2) {
    context.fillStyle = stockpile.type === "human" ? "#537d86" : "#69794b";
    context.beginPath();
    context.moveTo(px + 5, py + 9);
    context.lineTo(px + 12, py + 1);
    context.lineTo(px + 19, py + 9);
    context.fill();
    context.fillStyle = "#efe2bd";
    context.fillRect(px + 11, py + 5, 2, 4);
  }
  if (level >= 3) {
    const flame = tick % 2;
    context.fillStyle = "#68432e";
    context.fillRect(px - 2, py + 16, 7, 2);
    context.fillStyle = flame ? "#f2b33e" : "#e66c2f";
    context.fillRect(px, py + 12 - flame, 3, 4 + flame);
    context.fillStyle = "#ffe17a";
    context.fillRect(px + 1, py + 13 - flame, 1, 2);
  }
  if (food > 0) {
    context.fillStyle = "#fff3bd";
    context.font = "bold 7px monospace";
    context.fillText(String(food), px + 14, py + 7);
  }
}

function drawMatrixCell(context, value, x, y, size, matrix, path, tick) {
  const px = x * size;
  const py = y * size;
  const hash = (x * 37 + y * 61) % 97;
  context.fillStyle = path ? ((x + y) % 2 ? "#bfa97d" : "#c7b489") : (hash % 3 === 0 ? "#d5dfbd" : "#cfdbb7");
  context.fillRect(px, py, size, size);
  if (path) {
    context.fillStyle = "rgba(105,78,48,.13)";
    context.fillRect(px + (hash % 13), py + 5 + (hash % 9), 3, 1);
  } else {
    context.fillStyle = "#78945d";
    if (hash % 7 === 0) {
      context.fillRect(px + 4, py + 12, 1, 4);
      context.fillRect(px + 6, py + 13, 1, 3);
    }
    if (hash === 11 || hash === 47) {
      context.fillStyle = hash === 11 ? "#e7a1b4" : "#f1d26b";
      context.fillRect(px + 13, py + 7, 2, 2);
      context.fillStyle = "#567b4d";
      context.fillRect(px + 14, py + 9, 1, 3);
    }
    if (hash === 23 || hash === 71) {
      context.fillStyle = "#8b9588";
      context.fillRect(px + 2, py + 15, 4, 2);
      context.fillStyle = "#b7bdb1";
      context.fillRect(px + 3, py + 14, 2, 1);
    }
  }
  if (value === 1) {
    context.fillStyle = "rgba(38,49,43,.18)";
    context.fillRect(px + 3, py + 16, 15, 3);
    context.fillStyle = "#626d65";
    context.fillRect(px + 2, py + 8, 15, 9);
    context.fillRect(px + 5, py + 4, 10, 12);
    context.fillStyle = "#929b90";
    context.fillRect(px + 6, py + 5, 7, 3);
    context.fillRect(px + 3, py + 10, 4, 2);
    context.fillStyle = "#48524d";
    context.fillRect(px + 12, py + 13, 4, 3);
  } else if (value === 2) {
    context.fillStyle = "rgba(55,48,31,.18)";
    context.fillRect(px + 3, py + 17, 15, 2);
    context.fillStyle = "#3f7745";
    context.fillRect(px + 9, py + 6, 2, 12);
    context.fillRect(px + 4, py + 9, 12, 7);
    context.fillStyle = "#75a950";
    context.fillRect(px + 3, py + 7, 6, 5);
    context.fillRect(px + 11, py + 5, 6, 6);
    context.fillStyle = "#d94f4b";
    context.fillRect(px + 6, py + 11, 3, 3);
    context.fillRect(px + 12, py + 9, 3, 3);
    context.fillStyle = "#ffd675";
    context.fillRect(px + 7, py + 11, 1, 1);
  } else if (value === 3) {
    const wave = (tick + x * 3 + y * 5) % 10;
    context.fillStyle = wave < 5 ? "#388fb2" : "#3285aa";
    context.fillRect(px, py, size, size);
    context.fillStyle = "#75c7d9";
    context.fillRect(px + (wave % 4), py + 5, 8, 2);
    context.fillRect(px + 10 - (wave % 3), py + 14, 7, 1);
    context.fillStyle = "#b9dcce";
    if (y === 0 || matrix[y - 1][x] !== 3) context.fillRect(px, py, size, 2);
    if (x === 0 || matrix[y][x - 1] !== 3) context.fillRect(px, py, 2, size);
    if (y === matrix.length - 1 || matrix[y + 1][x] !== 3) context.fillRect(px, py + size - 1, size, 1);
    if (x === matrix[y].length - 1 || matrix[y][x + 1] !== 3) context.fillRect(px + size - 1, py, 1, size);
  }
}

function drawPixelAgent(context, agent, size, selected, tick, detailed) {
  const offset = spriteOffset(agent.id, detailed);
  const px = agent.x * size + offset.x;
  const py = agent.y * size + offset.y;
  if (selected) {
    const cellX = agent.x * size;
    const cellY = agent.y * size;
    context.fillStyle = "rgba(255,243,145,.34)";
    context.fillRect(cellX, cellY, size, size);
    context.strokeStyle = "#fff065";
    context.lineWidth = 2;
    context.strokeRect(cellX + 1, cellY + 1, size - 2, size - 2);
    context.fillStyle = "#fff065";
    context.fillRect(cellX + 8, cellY, 4, 2);
  }
  if (!agent.alive) {
    drawPixelCorpse(context, agent, px, py);
    return;
  }
  const moving = MOVEMENT_ACTIONS.has(agent.action);
  const frame = moving ? tick % 2 : 0;
  if (agent.dependent_ticks_remaining > 0) {
    drawPixelBaby(context, agent, px, py, frame);
    drawAgentMarkers(context, agent, px, py, tick, selected);
    return;
  }
  if (agent.type === "human") {
    drawPixelHuman(context, agent, px, py, frame, detailed);
  } else {
    drawPixelAnimal(context, agent, px, py, frame, detailed);
  }
  drawActionEffect(context, agent, px, py, tick);
  drawAgentMarkers(context, agent, px, py, tick, selected);
}

function spriteOffset(id, detailed) {
  if (!detailed) return {x: 0, y: 0};
  const hash = [...id].reduce((total, char) => total + char.charCodeAt(0), 0);
  return {x: hash % 3 - 1, y: Math.floor(hash / 3) % 2};
}

function drawPixelHuman(context, agent, px, py, frame, detailed) {
  const torso = agent.sex === "F" ? "#d95f91" : "#3f82bd";
  const dark = agent.sex === "F" ? "#8d3e69" : "#285b89";
  context.fillStyle = "rgba(49,37,27,.22)";
  context.fillRect(px + 4, py + 17, 12, 2);
  context.fillStyle = "#efad73";
  context.fillRect(px + 7, py + 3, 7, 6);
  context.fillStyle = "#6d392d";
  context.fillRect(px + 7, py + 2, 7, 3);
  if (agent.sex === "F") {
    context.fillRect(px + 6, py + 4, 2, 6);
    context.fillRect(px + 13, py + 4, 2, 5);
  } else if (detailed) {
    context.fillRect(px + 9, py + 1, 4, 1);
  }
  context.fillStyle = "#3d2925";
  context.fillRect(px + 9, py + 6, 1, 1);
  context.fillRect(px + 12, py + 6, 1, 1);
  context.fillStyle = torso;
  context.fillRect(px + 7, py + 9, 7, 6);
  context.fillRect(px + 4, py + 10 + frame, 3, 2);
  context.fillRect(px + 14, py + 10 + (1 - frame), 3, 2);
  context.fillStyle = dark;
  context.fillRect(px + 7 + frame, py + 15, 3, 3);
  context.fillRect(px + 11 - frame, py + 15, 3, 3);
  context.fillStyle = "#e7d7ae";
  context.fillRect(px + 8, py + 10, 1, 4);
}

function drawPixelAnimal(context, agent, px, py, frame, detailed) {
  context.fillStyle = "rgba(43,44,29,.22)";
  context.fillRect(px + 2, py + 16, 16, 2);
  if (agent.predator) {
    context.fillStyle = "#87453e";
    context.fillRect(px + 4, py + 8, 11, 7);
    context.fillRect(px + 13, py + 6, 6, 7);
    context.fillRect(px + 1, py + 7, 5, 3);
    context.fillStyle = "#5d302f";
    context.fillRect(px + 14, py + 3, 2, 4);
    context.fillRect(px + 17, py + 4, 2, 3);
    context.fillRect(px + 5 + frame, py + 14, 2, 4);
    context.fillRect(px + 13 - frame, py + 14, 2, 4);
    context.fillStyle = "#d78859";
    context.fillRect(px + 7, py + 9, 5, 2);
    context.fillStyle = "#fff0c7";
    context.fillRect(px + 17, py + 9, 2, 1);
    if (detailed) context.fillRect(px + 16, py + 12, 1, 2);
  } else {
    context.fillStyle = "#9b713f";
    context.fillRect(px + 4, py + 9, 11, 6);
    context.fillRect(px + 13, py + 7, 5, 6);
    context.fillStyle = "#6b4c31";
    context.fillRect(px + 14, py + 3, 2, 5);
    context.fillRect(px + 17, py + 4, 2, 4);
    context.fillRect(px + 5 + frame, py + 14, 2, 4);
    context.fillRect(px + 12 - frame, py + 14, 2, 4);
    context.fillStyle = "#d7b36c";
    context.fillRect(px + 6, py + 10, 6, 2);
    context.fillStyle = "#f7f1ce";
    context.fillRect(px + 16, py + 8, 1, 1);
    context.fillRect(px + 3, py + 9, 2, 2);
  }
  context.fillStyle = agent.sex === "F" ? "#e578a4" : "#65a5d7";
  context.fillRect(px + 13, py + 12, 3, 1);
}

function drawPixelBaby(context, agent, px, py, frame) {
  const color = agent.type === "human"
    ? (agent.sex === "F" ? "#e782ad" : "#6da5d6")
    : (agent.predator ? "#a95d4e" : "#b88d55");
  context.fillStyle = "rgba(48,42,29,.18)";
  context.fillRect(px + 6, py + 16, 9, 2);
  context.fillStyle = color;
  context.fillRect(px + 7, py + 10 - frame, 8, 6);
  context.fillStyle = agent.type === "human" ? "#f1b47d" : color;
  context.fillRect(px + 9, py + 6 - frame, 5, 5);
  context.fillStyle = "#332824";
  context.fillRect(px + 12, py + 8 - frame, 1, 1);
}

function drawPixelCorpse(context, agent, px, py) {
  context.fillStyle = "rgba(48,43,37,.18)";
  context.fillRect(px + 2, py + 16, 17, 2);
  if (agent.type === "human") {
    context.fillStyle = agent.sex === "F" ? "#89536b" : "#526b7c";
    context.fillRect(px + 5, py + 12, 11, 4);
    context.fillStyle = "#b68d72";
    context.fillRect(px + 15, py + 10, 4, 5);
  } else {
    context.fillStyle = agent.predator ? "#623b38" : "#74614a";
    context.fillRect(px + 3, py + 12, 14, 4);
    context.fillRect(px + 15, py + 10, 4, 5);
  }
  context.fillStyle = "#eee6d5";
  context.fillRect(px + 16, py + 11, 1, 1);
  context.fillRect(px + 18, py + 13, 1, 1);
}

function drawActionEffect(context, agent, px, py, tick) {
  if (agent.action === "ATTACK") {
    context.fillStyle = tick % 2 ? "#fff3c4" : "#e85e4f";
    context.fillRect(px + 16, py + 4, 2, 2);
    context.fillRect(px + 14, py + 6, 2, 2);
    context.fillRect(px + 18, py + 2, 1, 2);
  } else if (agent.action === "GATHER") {
    context.fillStyle = "#805633";
    context.fillRect(px + 1, py + 12, 5, 4);
    context.fillStyle = "#72a84f";
    context.fillRect(px + 2, py + 10, 3, 3);
  } else if (agent.action === "EAT") {
    context.fillStyle = "#e1b949";
    context.fillRect(px + 15, py + 7, 2, 2);
    context.fillRect(px + 17, py + 5, 1, 1);
  } else if (agent.action === "DRINK") {
    context.fillStyle = "#66c2df";
    context.fillRect(px + 16, py + 7, 2, 3);
    context.fillRect(px + 17, py + 6, 1, 1);
  } else if (agent.action === "REST") {
    context.fillStyle = "#5d6c79";
    context.font = "bold 7px monospace";
    context.fillText("z", px + 15, py + 5);
  }
}

function drawAgentMarkers(context, agent, px, py, tick, selected) {
  if (agent.carried_food > 0) {
    context.fillStyle = "#765033";
    context.fillRect(px + 1, py + 3, 6, 5);
    context.fillStyle = "#82b552";
    context.fillRect(px + 2, py + 1, 4, 4);
  }
  if (agent.pregnancy_ticks_remaining > 0) {
    context.fillStyle = "#f2ca5b";
    context.fillRect(px + 11, py + 12, 5, 4);
    context.fillStyle = "#fff0a2";
    context.fillRect(px + 13, py + 12, 2, 2);
  }
  if (agent.heart_ticks_remaining > 0) {
    const pulse = tick % 2;
    context.fillStyle = "#d93668";
    context.fillRect(px + 8 - pulse, py, 3, 3);
    context.fillRect(px + 12 + pulse, py, 3, 3);
    context.fillRect(px + 7, py + 2, 9, 3);
    context.fillRect(px + 9, py + 5, 5, 2);
    context.fillRect(px + 11, py + 7, 1, 1);
    context.fillStyle = "#ff91ad";
    context.fillRect(px + 9, py + 1, 1, 1);
  }
  if (agent.hunger >= 0.50) {
    const width = Math.max(1, Math.round(Math.min(1, agent.hunger) * 14));
    context.fillStyle = "rgba(49,38,30,.55)";
    context.fillRect(px + 3, py + 19, 14, 1);
    context.fillStyle = agent.hunger >= 0.70 ? "#d94f45" : "#e49a42";
    context.fillRect(px + 3, py + 19, width, 1);
  }
  if (selected) {
    context.fillStyle = "#fff7af";
    context.fillRect(px + 9, py - 2, 4, 2);
  }
}

function renderInspector() {
  let individual = selectedAgentId
    ? state.agents.find(item => item.id === selectedAgentId)
    : null;
  if (selectedAgentId && !individual) selectedAgentId = null;
  const selected = individual
    ? [individual]
    : state.agents.filter(item => item.type === selectedScope);
  ui["all-humans-button"].classList.toggle("active", !selectedAgentId && selectedScope === "human");
  ui["all-animals-button"].classList.toggle("active", !selectedAgentId && selectedScope === "animal");
  if (!selected.length) {
    ui["agent-empty"].hidden = false;
    ui["group-overview"].hidden = true;
    ui["agent-details"].hidden = true;
    return;
  }
  const view = aggregateAgents(selected);
  ui["agent-empty"].hidden = true;
  ui["agent-name"].textContent = individual
    ? individual.id
    : selectedScope === "human" ? "Todos los humanos" : "Todos los animales";
  ui["agent-kind"].textContent = individual
    ? `${individual.type} ${individual.sex}${individual.predator ? " · depredador" : ""} · ${individual.alive ? "vivo" : "muerto"}`
    : `grupo · ${selected.filter(agent => agent.alive).length}/${selected.length} vivos`;
  ui["brain-view-label"].textContent = individual
    ? `${individual.id} · ${individual.alive ? "vivo" : "muerto"}`
    : `${selectedScope === "human" ? "Humanos" : "Animales"} · promedio de ${selected.length} brains`;
  drawBrainNetwork(view, individual ? individual.id : `all-${selectedScope}`);

  if (!individual) {
    ui["group-overview"].hidden = false;
    ui["agent-details"].hidden = true;
    renderGroupOverview(selected, view);
    return;
  }

  ui["group-overview"].hidden = true;
  ui["agent-details"].hidden = false;
  renderRpgProfile(individual);
  const vitals = [
    ["Salud", view.health, false], ["Energía", view.energy, false],
    ["Hambre", view.hunger, true], ["Sed", view.thirst, true]
  ];
  ui.vitals.innerHTML = vitals.map(([name, value, inverse]) => `
    <div class="vital-row"><span>${name}</span><div class="vital-track"><div class="vital-fill ${inverse ? "warning" : ""}" style="width:${value * 100}%"></div></div><b>${format(value, 2)}</b></div>
  `).join("");
  ui["agent-action"].textContent = actionLabel(view.action);
  ui["agent-reward"].textContent = signed(view.reward, 3);
  ui["agent-loss"].textContent = view.loss == null ? "—" : format(view.loss, 4);
  ui["agent-mode"].textContent = individual
    ? `${individual.exploration_profile === "scout" ? "Explorador Horde" : "Perfil normal"} · ${view.exploration > 0 ? "explora ahora" : "decide el brain"} · ε ${format(view.epsilon, 2)}${individual.governor_override ? ` · gobernador cambió ${individual.brain_preferred_action} → ${individual.action}` : ""}${individual.survival_priority ? ` · prioridad ${individual.survival_priority}` : ""}`
    : `${view.scouts}/${selected.length} exploradores · ${format(view.exploration * 100, 0)}% explorando ahora · ε̄ ${format(view.epsilon, 2)} · gobernador interviene en ${format(view.governor_override * 100, 0)}%`;
  ui["epsilon-explanation"].textContent = `ε ${format(view.epsilon, 2)} significa cerca de ${format(view.epsilon * 100, 0)}% de probabilidad de probar una acción aleatoria. Es un rasgo individual persistente: perfiles normales 1–15%; exploradores Horde 50%.`;
  ui["agent-replay"].textContent = individual
    ? `Personal ${view.replay_size} · Horde ${view.horde_replay_size}`
    : `Horde ${view.horde_replay_size} · personal medio ${format(view.replay_size, 0)}`;
  ui["agent-updates"].textContent = individual ? `${view.training_steps} updates` : `${format(view.training_steps, 0)} updates medios`;

  renderSpatialMemory(selected, individual);

  const qValues = view.q_values;
  const extent = Math.max(...qValues.map(Math.abs), .001);
  ui["q-values"].innerHTML = ACTIONS.map((action, index) => `
    <div class="bar-row ${action === view.action ? "chosen" : ""}"><span>${action}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.abs(qValues[index] || 0) / extent * 100}%"></div></div><b>${format(qValues[index] || 0, 3)}</b></div>
  `).join("");

  ui.observations.innerHTML = view.observation_labels.map((label, index) => `<div class="observation"><span>${label}</span><b>${format(view.observation[index] || 0, 3)}</b></div>`).join("");
  const components = Object.entries(view.reward_components);
  ui["reward-components"].innerHTML = components.length ? components.map(([name, value]) => `<span class="component ${value > 0 ? "positive" : value < 0 ? "negative" : ""}">${name} ${signed(value, 2)}</span>`).join("") : `<span class="component">Sin acción todavía</span>`;
}

function renderGroupOverview(agents, view) {
  drawGroupChart(agents, view);
  const favorite = favoriteAction(view.action_counts);
  const facts = [
    ["Población", `${agents.filter(agent => agent.alive).length} / ${agents.length}`],
    ["Actividad favorita", actionLabel(favorite)],
    ["Reward promedio", signed(view.total_reward, 2)],
    ["Updates medios", format(view.training_steps, 0)],
  ];
  if (selectedScope === "human") {
    facts.push(["Principal causa de muerte", principalDeathCause(agents), "death-cause"]);
  }
  ui["group-summary"].innerHTML = facts.map(([label, value, kind]) => `<div class="group-stat ${kind || ""}"><span>${label}</span><strong>${value}</strong></div>`).join("");
}

function principalDeathCause(agents) {
  const causes = agents
    .filter(agent => !agent.alive && agent.cause_of_death)
    .map(agent => deathCauseCategory(agent.cause_of_death));
  if (!causes.length) return "Sin muertes registradas";
  const counts = new Map();
  for (const cause of causes) counts.set(cause, (counts.get(cause) || 0) + 1);
  const [cause, count] = [...counts.entries()]
    .sort((first, second) => second[1] - first[1] || first[0].localeCompare(second[0]))[0];
  return `${cause} · ${count} ${count === 1 ? "caso" : "casos"}`;
}

function deathCauseCategory(cause) {
  if (cause.startsWith("attack:") || cause.startsWith("predator_attack:")) {
    return "Ataque de depredador";
  }
  return deathCauseLabel(cause);
}

function drawGroupChart(agents, view) {
  const canvas = ui["group-chart"];
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#f5f4ee";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#17201c";
  context.font = "600 14px Georgia";
  context.fillText("Estado promedio", 18, 25);
  context.fillStyle = "#6d7771";
  context.font = "10px system-ui";
  context.fillText(`${agents.filter(agent => agent.alive).length} vivos de ${agents.length}`, 265, 25);
  const rows = [
    ["Salud", view.health, "#277657"],
    ["Energía", view.energy, "#3b7ea1"],
    ["Hambre", view.hunger, "#d97338"],
    ["Sed", view.thirst, "#7b68a6"],
  ];
  rows.forEach(([label, value, color], index) => {
    const y = 54 + index * 36;
    context.fillStyle = "#6d7771";
    context.font = "10px system-ui";
    context.fillText(label, 18, y);
    context.fillStyle = "#e1dfd6";
    context.fillRect(78, y - 9, 244, 10);
    context.fillStyle = color;
    context.fillRect(78, y - 9, 244 * Math.max(0, Math.min(1, value)), 10);
    context.fillStyle = "#17201c";
    context.font = "600 9px system-ui";
    context.textAlign = "right";
    context.fillText(`${format(value * 100, 0)}%`, 342, y);
    context.textAlign = "left";
  });
  const favorite = favoriteAction(view.action_counts);
  context.fillStyle = "#e4ebe4";
  context.fillRect(18, 202, 324, 31);
  context.fillStyle = "#185a45";
  context.font = "700 9px system-ui";
  context.fillText("ACTIVIDAD DOMINANTE", 28, 221);
  context.textAlign = "right";
  context.font = "600 11px system-ui";
  context.fillText(actionLabel(favorite), 330, 221);
  context.textAlign = "left";
}

function renderRpgProfile(agent) {
  const favorite = favoriteAction(agent.action_counts);
  const pronoun = agent.sex === "F" ? "Ella" : "Él";
  const role = agent.dependent_ticks_remaining > 0
    ? (agent.type === "human" ? "Bebé humano" : "Cría animal")
    : agent.type === "human"
    ? `${agent.sex === "F" ? "Humana" : "Humano"} ${agent.exploration_profile === "scout" ? (agent.sex === "F" ? "exploradora" : "explorador") : "habitante"}`
    : agent.predator ? "Animal depredador" : "Animal recolector";
  const strongestNeed = [["hambre", agent.hunger], ["sed", agent.thirst], ["cansancio", 1 - agent.energy]].sort((a, b) => b[1] - a[1])[0];
  const condition = !agent.alive
    ? "Su aventura terminó."
    : strongestNeed[1] >= .5
      ? `Ahora enfrenta ${strongestNeed[0]}.`
      : "Ahora se encuentra en condición estable.";
  ui["rpg-avatar"].textContent = agent.type === "human" ? "H" : "A";
  ui["rpg-avatar"].className = `rpg-avatar ${agent.type === "animal" ? "animal" : ""} ${agent.sex === "F" ? "female" : ""} ${agent.predator ? "predator" : ""}`;
  ui["rpg-title"].textContent = `${role} · ${agent.alive ? "nivel activo" : "caído"}`;
  ui["favorite-action"].textContent = `Actividad favorita: ${actionLabel(favorite)}`;
  const family = agent.heart_ticks_remaining > 0
    ? ` Comparte un vínculo con ${agent.heart_partner_id}.`
    : agent.pregnancy_ticks_remaining > 0
      ? ` Su embarazo termina en ${agent.pregnancy_ticks_remaining} ticks.`
      : agent.dependent_ticks_remaining > 0
        ? ` Sigue a ${agent.mother_id} durante ${agent.dependent_ticks_remaining} ticks más.`
        : agent.dependent_ids?.length
          ? ` Cuida ${agent.dependent_ids.length} ${agent.dependent_ids.length === 1 ? "bebé" : "bebés"}.`
          : "";
  ui["agent-story"].textContent = `${pronoun} ha sobrevivido ${agent.steps_survived} ticks y acumulado ${signed(agent.total_reward, 2)} de reward. ${condition}${family}`;
  ui["rpg-facts"].innerHTML = [
    ["Sexo", agent.sex],
    ["Posición", `${agent.x}, ${agent.y}`],
    ["Edad", `${agent.steps_survived} ticks`],
    ...(agent.type === "human" ? [["Hijos", agent.children_born || 0]] : []),
    ...(agent.carried_food > 0 ? [["Carga", `${agent.carried_food} comida`]] : []),
    ...(agent.heart_ticks_remaining > 0 ? [["Corazón", `${agent.heart_ticks_remaining} ticks`]] : []),
    ...(agent.pregnancy_ticks_remaining > 0 ? [["Embarazo", `${agent.pregnancy_ticks_remaining} ticks`]] : []),
    ...(agent.dependent_ids?.length ? [["Bebés", agent.dependent_ids.length]] : []),
    ...(agent.alive ? [] : [["Causa de muerte", deathCauseLabel(agent.cause_of_death), "death"]]),
  ].map(([label, value, kind]) => `<div class="rpg-fact ${kind || ""}"><span>${label}</span><strong>${value}</strong></div>`).join("");
}

function deathCauseLabel(cause) {
  if (!cause) return "No registrada";
  if (cause.startsWith("attack:") || cause.startsWith("predator_attack:")) {
    return `Ataque de ${cause.split(":", 2)[1]}`;
  }
  const labels = {
    starvation: "Hambre",
    dehydration: "Deshidratación",
    exhaustion: "Agotamiento",
    unknown: "Desconocida",
  };
  return cause.split("+").map(item => labels[item] || item).join(" + ");
}

function favoriteAction(counts = {}) {
  const ranked = ACTIONS
    .map(action => [action, Number(counts[action] || 0)])
    .sort((first, second) => second[1] - first[1]);
  return ranked[0]?.[1] > 0 ? ranked[0][0] : null;
}

function actionLabel(action) {
  return ({
    MOVE_UP: "Moverse arriba", MOVE_DOWN: "Moverse abajo",
    MOVE_LEFT: "Moverse a la izquierda", MOVE_RIGHT: "Moverse a la derecha",
    EAT: "Comer", DRINK: "Beber", REST: "Descansar", WAIT: "Esperar",
    ATTACK: "Atacar", GATHER: "Recolectar", MATE: "Aparearse",
    FOLLOW_MOTHER: "Seguir a su madre", WAITING: "Esperando", DEAD: "Muerto",
  })[action] || "Sin historial";
}

function aggregateAgents(agents) {
  const first = agents[0];
  const mean = key => agents.reduce((sum, agent) => sum + Number(agent[key] || 0), 0) / agents.length;
  const meanNullable = key => {
    const values = agents.map(agent => agent[key]).filter(value => value != null);
    return values.length ? values.reduce((sum, value) => sum + Number(value), 0) / values.length : null;
  };
  const averageVectors = vectors => {
    const width = Math.max(0, ...vectors.map(vector => vector.length));
    return Array.from({length: width}, (_, index) => vectors.reduce((sum, vector) => sum + Number(vector[index] || 0), 0) / vectors.length);
  };
  const actionCounts = new Map();
  for (const agent of agents) actionCounts.set(agent.action, (actionCounts.get(agent.action) || 0) + 1);
  const action = [...actionCounts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || "WAITING";
  const historicActionCounts = Object.fromEntries(ACTIONS.map(action => [
    action,
    agents.reduce((sum, agent) => sum + Number(agent.action_counts?.[action] || 0), 0),
  ]));
  const componentNames = new Set(agents.flatMap(agent => Object.keys(agent.reward_components || {})));
  const rewardComponents = Object.fromEntries([...componentNames].map(name => [name, agents.reduce((sum, agent) => sum + Number(agent.reward_components?.[name] || 0), 0) / agents.length]));
  const activationNames = new Set(agents.flatMap(agent => Object.keys(agent.brain_activations || {})));
  const activations = Object.fromEntries([...activationNames].map(name => [name, averageVectors(agents.map(agent => agent.brain_activations?.[name] || []))]));
  const layerNames = first.weight_statistics?.map(row => row.layer) || [];
  const weights = layerNames.map(layer => {
    const rows = agents.map(agent => agent.weight_statistics.find(row => row.layer === layer)).filter(Boolean);
    return {layer, mean_abs: rows.reduce((sum, row) => sum + row.mean_abs, 0) / rows.length, max_abs: rows.reduce((sum, row) => sum + row.max_abs, 0) / rows.length};
  });
  return {
    health: mean("health"), energy: mean("energy"), hunger: mean("hunger"), thirst: mean("thirst"),
    total_reward: mean("total_reward"), steps_survived: mean("steps_survived"),
    reward: mean("reward"), loss: meanNullable("loss"), epsilon: mean("epsilon"),
    exploration: agents.filter(agent => agent.exploration).length / agents.length,
    governor_override: agents.filter(agent => agent.governor_override).length / agents.length,
    scouts: agents.filter(agent => agent.exploration_profile === "scout").length,
    replay_size: mean("replay_size"), horde_replay_size: mean("horde_replay_size"),
    training_steps: mean("training_steps"), action, action_counts: historicActionCounts,
    q_values: averageVectors(agents.map(agent => agent.q_values || [])),
    observation: averageVectors(agents.map(agent => agent.observation || [])),
    observation_labels: [...(first.need_labels || []), ...(first.spatial_labels || [])],
    brain_architecture: first.brain_architecture, brain_activations: activations,
    weight_statistics: weights, reward_components: rewardComponents,
  };
}

function renderSpatialMemory(agents, individual) {
  if (individual) {
    ui["spatial-memory"].innerHTML = ["food", "water"].map(resource => {
      const memory = individual.spatial_memory?.[resource] || {};
      const label = resource === "food" ? "Comida" : "Agua";
      const place = memory.position ? `(${memory.position.join(", ")})` : "no vista";
      const age = memory.age == null ? "—" : `${memory.age} ticks`;
      return `<div class="memory-card"><strong>${label}</strong><span>${place}</span><small>edad ${age} · confianza ${format((memory.confidence || 0) * 100, 0)}%</small></div>`;
    }).join("");
    return;
  }
  ui["spatial-memory"].innerHTML = ["food", "water"].map(resource => {
    const remembered = agents.filter(agent => agent.spatial_memory?.[resource]?.position).length;
    const confidence = agents.reduce((sum, agent) => sum + Number(agent.spatial_memory?.[resource]?.confidence || 0), 0) / agents.length;
    return `<div class="memory-card"><strong>${resource === "food" ? "Comida" : "Agua"}</strong><span>${remembered}/${agents.length} recuerdan</span><small>confianza media ${format(confidence * 100, 0)}%</small></div>`;
  }).join("");
}

function drawBrainNetwork(view, viewKey) {
  const canvas = ui["brain-network"];
  const context = canvas.getContext("2d");
  const architecture = view.brain_architecture || {};
  const activations = view.brain_activations || {};
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#f4f3ed";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#6d7771";
  context.font = "800 10px system-ui";
  context.fillText("1 · PERCIBIR", 32, 28);
  context.fillText("2 · INTERPRETAR", 298, 28);
  context.fillText("3 · INTEGRAR", 584, 28);
  context.fillText("4 · DECIDIR", 880, 28);

  const cards = {
    needInputs: {x: 28, y: 48, w: 210, h: 125, color: "#d97338", values: activations.need_inputs || []},
    spatialInputs: {x: 28, y: 218, w: 210, h: 125, color: "#3b7ea1", values: activations.spatial_inputs || []},
    needHidden: {x: 294, y: 48, w: 210, h: 125, color: "#df8a42", values: activations.need_hidden || []},
    spatialHidden: {x: 294, y: 218, w: 210, h: 125, color: "#4e98b6", values: activations.spatial_hidden || []},
    fusion: {x: 580, y: 104, w: 230, h: 184, color: "#277657", values: activations.fusion_hidden || []},
    actions: {x: 876, y: 42, w: 414, h: 326, color: "#78588f", values: view.q_values || []},
  };
  drawFlowConnector(context, cards.needInputs, cards.needHidden, activationLevel(cards.needInputs.values));
  drawFlowConnector(context, cards.spatialInputs, cards.spatialHidden, activationLevel(cards.spatialInputs.values));
  drawFlowConnector(context, cards.needHidden, cards.fusion, activationLevel(cards.needHidden.values));
  drawFlowConnector(context, cards.spatialHidden, cards.fusion, activationLevel(cards.spatialHidden.values));
  drawFlowConnector(context, cards.fusion, cards.actions, activationLevel(cards.fusion.values));

  drawStageCard(context, cards.needInputs, "Necesidades", `${architecture.need_input_size || 0} señales vitales`);
  drawStageCard(context, cards.spatialInputs, "Mundo y memoria", `${architecture.spatial_input_size || 0} señales espaciales`);
  drawStageCard(context, cards.needHidden, "Prioridad vital", `${architecture.hidden_sizes?.[0] || 0} neuronas`);
  drawStageCard(context, cards.spatialHidden, "Mapa interno", `${architecture.hidden_sizes?.[1] || 0} neuronas`);
  drawStageCard(context, cards.fusion, "Fusión", `${architecture.hidden_sizes?.[2] || 0} neuronas combinan ambas rutas`);
  drawActionCard(context, cards.actions, view);

  const currentMeans = Object.fromEntries(view.weight_statistics.map(row => [row.layer, row.mean_abs]));
  const previous = previousWeightMeans.get(viewKey);
  const averageWeight = Object.values(currentMeans).reduce((sum, value) => sum + value, 0) / Math.max(1, Object.values(currentMeans).length);
  const previousAverage = previous ? Object.values(previous).reduce((sum, value) => sum + value, 0) / Math.max(1, Object.values(previous).length) : null;
  const delta = previousAverage == null ? null : averageWeight - previousAverage;
  previousWeightMeans.set(viewKey, currentMeans);
  ui["brain-network-caption"].textContent = `Las dos rutas leen necesidades y mundo social por separado; Fusión reúne ambas antes de producir once Q-values. Acción actual: ${actionLabel(view.action)}. Fuerza media de pesos ${format(averageWeight, 5)}${delta == null ? "" : ` · cambio ${signed(delta, 7)}`}.`;
}

function drawStageCard(context, card, title, detail) {
  roundedRectPath(context, card.x, card.y, card.w, card.h, 14);
  context.fillStyle = "rgba(255,255,255,.82)";
  context.fill();
  context.strokeStyle = "#d8d8cf";
  context.lineWidth = 1;
  context.stroke();
  context.fillStyle = card.color;
  context.fillRect(card.x, card.y + 14, 5, card.h - 28);
  context.fillStyle = "#17201c";
  context.font = "650 15px Georgia";
  context.fillText(title, card.x + 18, card.y + 28);
  context.fillStyle = "#6d7771";
  context.font = "10px system-ui";
  context.fillText(detail, card.x + 18, card.y + 46);
  const values = card.values.length ? card.values : [0];
  const shown = Math.min(10, values.length);
  for (let index = 0; index < shown; index += 1) {
    const source = values.length <= shown ? index : Math.round(index * (values.length - 1) / Math.max(1, shown - 1));
    const strength = Math.min(1, Math.abs(Number(values[source] || 0)));
    const x = card.x + 20 + index * ((card.w - 40) / Math.max(1, shown - 1));
    const y = card.y + card.h - 34;
    context.globalAlpha = .25 + strength * .75;
    context.fillStyle = card.color;
    context.beginPath();
    context.arc(x, y, 4 + strength * 4, 0, Math.PI * 2);
    context.fill();
  }
  context.globalAlpha = 1;
  const level = activationLevel(values);
  context.fillStyle = "#e5e3da";
  context.fillRect(card.x + 18, card.y + card.h - 15, card.w - 36, 4);
  context.fillStyle = card.color;
  context.fillRect(card.x + 18, card.y + card.h - 15, (card.w - 36) * level, 4);
}

function drawActionCard(context, card, view) {
  roundedRectPath(context, card.x, card.y, card.w, card.h, 14);
  context.fillStyle = "rgba(255,255,255,.88)";
  context.fill();
  context.strokeStyle = "#d8d8cf";
  context.stroke();
  context.fillStyle = "#17201c";
  context.font = "650 15px Georgia";
  context.fillText("Acciones posibles", card.x + 18, card.y + 25);
  context.fillStyle = "#6d7771";
  context.font = "10px system-ui";
  context.fillText("Q-value aprendido para cada decisión", card.x + 18, card.y + 42);
  const values = view.q_values || [];
  const extent = Math.max(.001, ...values.map(value => Math.abs(Number(value || 0))));
  ACTIONS.forEach((action, index) => {
    const value = Number(values[index] || 0);
    const y = card.y + 66 + index * 27;
    const chosen = action === view.action;
    if (chosen) {
      roundedRectPath(context, card.x + 10, y - 15, card.w - 20, 23, 6);
      context.fillStyle = "#fff0dc";
      context.fill();
    }
    context.fillStyle = chosen ? "#9a541f" : "#536159";
    context.font = chosen ? "750 10px system-ui" : "550 9px system-ui";
    context.fillText(actionLabel(action), card.x + 18, y);
    context.fillStyle = "#e5e3da";
    context.fillRect(card.x + 158, y - 8, 170, 7);
    context.fillStyle = chosen ? "#d97338" : "#78588f";
    context.fillRect(card.x + 158, y - 8, 170 * Math.abs(value) / extent, 7);
    context.fillStyle = "#17201c";
    context.font = "600 9px ui-monospace, monospace";
    context.textAlign = "right";
    context.fillText(format(value, 3), card.x + card.w - 18, y);
    context.textAlign = "left";
  });
}

function drawFlowConnector(context, from, to, strength) {
  const startX = from.x + from.w;
  const startY = from.y + from.h / 2;
  const endX = to.x;
  const endY = to.y + to.h / 2;
  const bend = (endX - startX) * .52;
  context.strokeStyle = `rgba(39,118,87,${.18 + strength * .55})`;
  context.lineWidth = 2 + strength * 4;
  context.beginPath();
  context.moveTo(startX, startY);
  context.bezierCurveTo(startX + bend, startY, endX - bend, endY, endX - 8, endY);
  context.stroke();
  context.fillStyle = `rgba(39,118,87,${.35 + strength * .55})`;
  context.beginPath();
  context.moveTo(endX - 8, endY - 5);
  context.lineTo(endX, endY);
  context.lineTo(endX - 8, endY + 5);
  context.fill();
}

function activationLevel(values) {
  if (!values?.length) return 0;
  return Math.min(1, values.reduce((sum, value) => sum + Math.abs(Number(value || 0)), 0) / values.length);
}

function roundedRectPath(context, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + r, y);
  context.arcTo(x + width, y, x + width, y + height, r);
  context.arcTo(x + width, y + height, x, y + height, r);
  context.arcTo(x, y + height, x, y, r);
  context.arcTo(x, y, x + width, y, r);
  context.closePath();
}

function renderEvents() {
  const events = (state.recent_events || []).slice(0, 24);
  ui["event-stream"].innerHTML = events.length ? events.map(event => `
    <div class="event"><span class="tick">#${event.tick}</span><strong>${event.agent_id}<br>${actionLabel(event.action)}</strong><span class="${event.trained ? "trained" : ""}">${event.trained ? `↻ ${format(event.loss, 4)}` : signed(event.reward, 2)}</span></div>
  `).join("") : `<div class="empty-state" style="min-height:100px">Inicia o avanza un tick.</div>`;
}

function renderTrainingState() {
  const trained = state.agents.filter(agent => agent.trained).length;
  const flowItems = document.querySelectorAll("#learning-flow span");
  flowItems.forEach(item => item.classList.toggle("active", state.status === "running" && (item.textContent !== "Backward" || trained > 0)));
  if (state.status === "completed") {
    const reason = state.termination_reason === "human_extinction"
      ? `Run detenido en el tick ${state.tick}: murió el último humano.`
      : state.termination_reason === "user_cancelled"
        ? `Experimento cancelado en el tick ${state.tick}.`
        : `Run terminado en el tick ${state.tick}.`;
    const champion = state.brb_promoted ? " Este run superó el campeón anterior y ahora es el nuevo BRB." : "";
    const persistence = state.result
      ? ` Checkpoints y métricas guardados en ${state.result.results_dir}.`
      : " No se guardó un checkpoint porque aún no había avanzado ningún tick.";
    ui["training-message"].textContent = `${reason}${champion}${persistence}`;
  } else if (state.tick === 0 && state.brb_source) {
    ui["training-message"].textContent = `Nuevo experimento desde BRB: ${state.agents.length} brains heredaron pesos de ${state.brb_source}. Adam, replay Horde, cuerpos y mundo comenzaron limpios.`;
  } else if (trained > 0) {
    ui["training-message"].textContent = `${trained} brains actualizaron sus pesos después de reunir todas las experiencias del tick en Horde. La loss proviene de backpropagation real.`;
  } else if (state.tick === 0 && state.continued_from) {
    const migration = state.architecture_migrations?.length
      ? ` ${state.architecture_migrations.length} brains fueron ensanchados conservando sus outputs iniciales.`
      : "";
    const rewardReset = state.learning_state_resets?.length
      ? ` Se limpiaron replay, Adam y target porque cambió la función de reward; se conservaron los perfiles epsilon individuales.`
      : "";
    ui["training-message"].textContent = `Nuevo ciclo con seed ${state.seed}: los ${state.agents.length} brains fueron cargados desde ${state.continued_from}.${migration}${rewardReset} El mundo físico comenzó de nuevo.`;
  } else {
    const humanHorde = state.horde?.human_replay_size || 0;
    const animalHorde = state.horde?.animal_replay_size || 0;
    ui["training-message"].textContent = `Horde comparte experiencias por especie: humanos ${humanHorde}, animales ${animalHorde}. Cada brain mantiene pesos propios y aprende de ese replay colectivo.`;
  }
}

function renderCompletionSummary() {
  const summary = state.learning_summary;
  ui["completion-summary"].hidden = !summary;
  if (!summary) {
    ui["comparison-chart-wrap"].hidden = true;
    return;
  }
  const extinct = summary.termination_reason === "human_extinction";
  const cancelled = summary.termination_reason === "user_cancelled";
  ui["termination-label"].textContent = extinct
    ? `Extinción humana · tick ${summary.ticks_executed}`
    : cancelled
      ? `Cancelado · tick ${summary.ticks_executed}`
      : `Límite alcanzado · tick ${summary.ticks_executed}`;
  ui["survivor-summary"].textContent = `Humanos: ${summary.initial_humans} → ${summary.final_humans} · Animales: ${summary.initial_animals} → ${summary.final_animals}`;
  const comparison = summary.comparison_to_previous_run;
  ui["run-comparison"].hidden = !comparison;
  if (comparison) {
    const change = comparison.last_human_tick_change;
    ui["run-comparison"].textContent = `Run anterior → actual: último humano ${comparison.previous_last_human_tick} → ${comparison.current_last_human_tick} ticks (${change >= 0 ? "+" : ""}${change}); supervivencia humana media ${format(comparison.previous_mean_human_survival, 1)} → ${format(comparison.current_mean_human_survival, 1)}.`;
  }
  drawRunComparison(comparison);
  ui["learning-numbers"].innerHTML = [["human", "Humanos"], ["animal", "Animales"]].map(([key, label]) => {
    const values = summary.by_type[key];
    return `<tr><td><strong>${label}</strong></td><td>${signed(values.first_20_percent_reward, 4)}</td><td>${signed(values.last_20_percent_reward, 4)}</td><td>${format(values.first_20_percent_loss, 4)}</td><td>${format(values.last_20_percent_loss, 4)}</td><td>${values.successful_drinks} (${values.brain_selected_drinks} brain)</td></tr>`;
  }).join("");
  ui["learned-explanation"].innerHTML = summary.what_they_learned.map(sentence => `<li>${sentence}</li>`).join("");
}

function drawRunComparison(comparison) {
  ui["comparison-chart-wrap"].hidden = !comparison;
  if (!comparison) return;
  const canvas = ui["comparison-chart"];
  const context = canvas.getContext("2d");
  const data = [
    {label: "Último humano", previous: comparison.previous_last_human_tick, current: comparison.current_last_human_tick},
    {label: "Supervivencia media", previous: comparison.previous_mean_human_survival, current: comparison.current_mean_human_survival}
  ];
  const maxValue = Math.max(1, ...data.flatMap(item => [item.previous, item.current]));
  const baseline = 210;
  const chartHeight = 150;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#f4f3ed";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#17201c";
  context.font = "600 13px system-ui";
  context.fillText(`Run ${pad(comparison.previous_run_number)} vs Run ${pad(state.run_number)}`, 28, 28);
  context.font = "11px system-ui";
  context.fillStyle = "#6d7771";
  context.fillText("Run anterior", 500, 27);
  context.fillStyle = "#9aa39c";
  context.fillRect(480, 18, 12, 12);
  context.fillStyle = "#6d7771";
  context.fillText("Run actual", 610, 27);
  context.fillStyle = "#d97338";
  context.fillRect(590, 18, 12, 12);
  data.forEach((item, index) => {
    const groupX = 115 + index * 310;
    const previousHeight = item.previous / maxValue * chartHeight;
    const currentHeight = item.current / maxValue * chartHeight;
    context.fillStyle = "#9aa39c";
    context.fillRect(groupX, baseline - previousHeight, 62, previousHeight);
    context.fillStyle = "#d97338";
    context.fillRect(groupX + 72, baseline - currentHeight, 62, currentHeight);
    context.textAlign = "center";
    context.font = "600 11px system-ui";
    context.fillStyle = "#17201c";
    context.fillText(format(item.previous, 1), groupX + 31, baseline - previousHeight - 7);
    context.fillText(format(item.current, 1), groupX + 103, baseline - currentHeight - 7);
    context.font = "11px system-ui";
    context.fillStyle = "#6d7771";
    context.fillText(item.label, groupX + 67, baseline + 22);
    context.textAlign = "start";
  });
}

function showError(message) {
  ui.toast.textContent = message;
  ui.toast.hidden = false;
  setTimeout(() => { ui.toast.hidden = true; }, 5000);
}
function statusText(status) {
  return ({paused: "PAUSADO", running: "ENTRENANDO", preparing: "PREPARANDO", finalizing: "GUARDANDO", completed: "COMPLETO", error: "ERROR"})[status] || status.toUpperCase();
}
function format(value, digits = 2) { return Number(value).toFixed(digits); }
function signed(value, digits = 2) { const number = Number(value); return `${number >= 0 ? "+" : ""}${number.toFixed(digits)}`; }
function pad(value) { return String(value).padStart(3, "0"); }

ui["play-button"].addEventListener("click", () => control("play"));
ui["pause-button"].addEventListener("click", () => control("pause"));
ui["step-button"].addEventListener("click", () => control("step"));
ui["cancel-button"].addEventListener("click", () => {
  if (window.confirm("¿Cancelar este experimento? Si ya avanzó, se guardará como run parcial.")) {
    control("cancel");
  }
});
ui["next-run-button"].addEventListener("click", () => control("next_run"));
ui["all-humans-button"].addEventListener("click", () => {
  selectedAgentId = null;
  selectedScope = "human";
  renderWorld();
  renderInspector();
});
ui["all-animals-button"].addEventListener("click", () => {
  selectedAgentId = null;
  selectedScope = "animal";
  renderWorld();
  renderInspector();
});
ui["speed-input"].addEventListener("input", event => { ui["speed-label"].textContent = `${event.target.value} ticks/s`; });
ui["speed-input"].addEventListener("change", event => control("speed", Number(event.target.value)));
for (const id of ["human-count-input", "animal-count-input", "human-brain-input", "animal-brain-input"]) {
  ui[id].addEventListener("input", () => {
    configurationDirty = true;
    updateBuilderLabels();
  });
}
ui["use-brb-input"].addEventListener("change", () => {
  configurationDirty = true;
  applyBrbMinimums();
  renderExperimentBuilder();
});
ui["apply-config-button"].addEventListener("click", () => {
  const settings = {
    num_humans: Number(ui["human-count-input"].value),
    num_animals: Number(ui["animal-count-input"].value),
    human_brain_width: Number(ui["human-brain-input"].value),
    animal_brain_width: Number(ui["animal-brain-input"].value),
    use_brb: ui["use-brb-input"].checked
  };
  selectedAgentId = null;
  selectedScope = "human";
  configurationDirty = false;
  control("new_experiment", settings);
});
ui.world.addEventListener("click", event => {
  if (!state) return;
  const bounds = ui.world.getBoundingClientRect();
  const gridX = (event.clientX - bounds.left) / bounds.width * state.grid.width;
  const gridY = (event.clientY - bounds.top) / bounds.height * state.grid.height;
  const x = Math.floor(gridX);
  const y = Math.floor(gridY);
  const candidates = state.agents.filter(agent => agent.x === x && agent.y === y);
  let agent = candidates.find(candidate => candidate.alive) || candidates[0];
  if (!agent) {
    const nearby = state.agents
      .map(candidate => ({
        candidate,
        distance: Math.hypot(candidate.x + .5 - gridX, candidate.y + .5 - gridY),
      }))
      .filter(item => item.distance <= 1.15)
      .sort((first, second) => {
        if (first.candidate.alive !== second.candidate.alive) return first.candidate.alive ? -1 : 1;
        return first.distance - second.distance;
      });
    agent = nearby[0]?.candidate;
  }
  if (agent) {
    selectedAgentId = agent.id;
    selectedScope = agent.type;
  } else {
    selectedAgentId = null;
  }
  renderWorld();
  renderInspector();
});

poll();
