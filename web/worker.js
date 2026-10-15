// Runs the Python simulator in Pyodide, off the main thread.
// Messages in:  {type: "reset", seed, scenario} | {type: "config", sensors} | {type: "speed", value} | {type: "pause", value}
// Messages out: {type: "status", text} | {type: "settings", data} | {type: "road", data} | {type: "frame", data} | {type: "error", text}

importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

const SIM_FILES = ["__init__.py", "world.py", "car.py", "traffic.py", "sensors.py", "env.py", "expert.py"];
const STEP_MS = 100; // the sim's 0.1 s step in real time

let py, bridge;
let speed = 1, paused = false, timer = null;

async function boot() {
  postMessage({ type: "status", text: "Loading Python…" });
  py = await loadPyodide();
  postMessage({ type: "status", text: "Loading numpy…" });
  await py.loadPackage("numpy");
  postMessage({ type: "status", text: "Loading simulator…" });
  py.FS.mkdir("sim");
  for (const f of SIM_FILES) py.FS.writeFile(`sim/${f}`, await (await fetch(`../sim/${f}`, { cache: "no-cache" })).text());
  py.FS.writeFile("bridge.py", await (await fetch("bridge.py", { cache: "no-cache" })).text());
  bridge = py.pyimport("bridge");
  postMessage({ type: "settings", data: bridge.settings() });
}

function loop() {
  clearTimeout(timer);
  if (!paused) postMessage({ type: "frame", data: bridge.step() });
  timer = setTimeout(loop, STEP_MS / speed);
}

const ready = boot().catch((e) => postMessage({ type: "error", text: String(e) }));

onmessage = async ({ data: msg }) => {
  await ready;
  if (!bridge) return;
  if (msg.type === "reset") {
    postMessage({ type: "road", data: bridge.reset(msg.seed, JSON.stringify(msg.scenario || {})) });
    loop();
  } else if (msg.type === "config") {
    bridge.configure(JSON.stringify(msg.sensors));
  } else if (msg.type === "speed") {
    speed = msg.value;
  } else if (msg.type === "pause") {
    paused = msg.value;
  }
};
