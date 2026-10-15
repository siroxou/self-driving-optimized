// 3D view and controls. The simulator runs in worker.js; this file only draws what it reports.
// Sim coordinates (x, y) on the ground map to three.js (x, -z), with y up.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const STEP_MS = 100;
const COLORS = { lidar: 0x4da3ff, sonar: 0xffa64d, radar: 0xffd84d, gps: 0x3ccf8e, ego: 0x2f7cf6, traffic: 0xd9dde3, parked: 0xff7a45 };
const LABELS = {
  odom: "Odometry + IMU", lane: "Lane camera", nav: "Route, exact position", gps: "Route, GPS",
  lidar: "Lidar", sonar: "Sonar (ultrasonic)", radar: "Radar",
};
const SETTING_LABELS = {
  noise: "noise", dropout: "dropout", beams: "beams", tracks: "tracks",
  range: "range (m)", jitter: "jitter (m)", drift: "drift (m/step)",
};
const SCENARIO_LABELS = {
  lanes: "lanes", complexity: "bendiness", traffic: "traffic", parked: "parked share",
  pedestrians: "crossings per km", obstacles: "debris per km", potholes: "potholes per km",
};
const SHORT = { odom: "Odometry", lane: "Camera", nav: "Route", gps: "GPS", lidar: "Lidar", sonar: "Sonar", radar: "Radar" };
const DRIVER_NOTES = {
  generalist: "One neural network trained with random sensor setups. It sees only the sensors switched on below, and copes with some setups better than others.",
  reference: "A hand-written lane follower that reads the simulator directly. It ignores the sensors, so switches change only what's drawn.",
  noSpecialist: "No specialist was trained on this exact sensor set, so the generalist is driving. Pick a set below to try one that has.",
  missing: "That model couldn't be loaded, so the reference lane follower is driving.",
};
// Specialists available for the Auto driver, listed by train/specialists.py.
const specialistsReady = fetch("models/specialists.json", { cache: "no-cache" })
  .then((r) => (r.ok ? r.json() : []))
  .catch(() => []);
const END_TEXT = {
  crash: "Crashed", out_of_road: "Left the road", success: "Arrived", timeout: "Out of time",
  hit_pedestrian: "Hit a pedestrian", damaged: "Damaged by a pothole",
};

const v3 = (x, y, h = 0) => new THREE.Vector3(x, h, -y);
const $ = (id) => document.getElementById(id);

// ---------- scene ----------
const renderer = new THREE.WebGLRenderer({ canvas: $("scene"), antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xa9c7e0);
scene.fog = new THREE.Fog(0xa9c7e0, 120, 420);
const camera = new THREE.PerspectiveCamera(60, 1, 0.5, 2000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enabled = false;
scene.add(new THREE.HemisphereLight(0xffffff, 0x4a5a3a, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(80, 150, 40);
scene.add(sun);

const ground = new THREE.Mesh(new THREE.PlaneGeometry(6000, 6000), new THREE.MeshLambertMaterial({ color: 0x6f8f55 }));
ground.rotation.x = -Math.PI / 2;
scene.add(ground);

let roadGroup = new THREE.Group();
scene.add(roadGroup);

function carMesh(color, opacity = 1) {
  const mat = new THREE.MeshLambertMaterial({ color, transparent: opacity < 1, opacity });
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(4.5, 0.8, 1.9), mat);
  body.position.y = 0.6;
  const cabin = new THREE.Mesh(new THREE.BoxGeometry(2.3, 0.65, 1.7), mat);
  cabin.position.set(-0.3, 1.32, 0);
  const glass = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.5, 1.6), new THREE.MeshLambertMaterial({ color: 0x1b2633, transparent: opacity < 1, opacity }));
  glass.position.set(0.86, 1.3, 0);
  g.add(body, cabin, glass);
  return g;
}

const ego = carMesh(COLORS.ego);
const ghost = carMesh(COLORS.gps, 0.5);
scene.add(ego, ghost);
const trafficMeshes = [], pedMeshes = [];
let potholeMeshes = [];

function pedMesh(i) {
  const g = new THREE.Group();
  const shirt = [0xd94f4f, 0x4f7fd9, 0xe0b341, 0x6bbf59, 0x9a5fd0][i % 5];
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.22, 0.85, 4, 8), new THREE.MeshLambertMaterial({ color: shirt }));
  body.position.y = 0.65;
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.15, 10, 8), new THREE.MeshLambertMaterial({ color: 0xe8c39e }));
  head.position.y = 1.45;
  g.add(body, head);
  return g;
}

function hazardMeshes(road, group) {
  const white = new THREE.MeshLambertMaterial({ color: 0xf2f2f2 });
  for (const [x, y, h] of road.crosswalks) {  // zebra stripes across the whole road
    const nx = -Math.sin(h), ny = Math.cos(h);
    for (let lat = -road.width / 2 + 0.6; lat < road.width / 2 - 0.3; lat += 1.1) {
      const stripe = new THREE.Mesh(new THREE.BoxGeometry(3.0, 0.02, 0.55), white);
      stripe.position.copy(v3(x + nx * lat, y + ny * lat, 0.04));
      stripe.rotation.y = h;
      group.add(stripe);
    }
  }
  const cone = new THREE.MeshLambertMaterial({ color: 0xff6a1a }), load = new THREE.MeshLambertMaterial({ color: 0x8a6a45 });
  for (const [x, y, h, size] of road.obstacles) {
    const m = size < 1
      ? new THREE.Mesh(new THREE.ConeGeometry(0.25, 0.75, 12), cone)
      : new THREE.Mesh(new THREE.BoxGeometry(size, 0.9, size), load);
    m.position.copy(v3(x, y, size < 1 ? 0.38 : 0.45));
    m.rotation.y = h + (size < 1 ? 0 : 0.4);
    group.add(m);
  }
  potholeMeshes = road.potholes.map(([x, y, r], i) => {
    const m = new THREE.Mesh(new THREE.CircleGeometry(r, 14), new THREE.MeshLambertMaterial({ color: 0x17181b }));
    m.rotation.x = -Math.PI / 2;
    m.scale.set(1 + 0.25 * Math.sin(i * 7.1), 1, 1);  // a little irregular
    m.position.copy(v3(x, y, 0.03));
    group.add(m);
    return m;
  });
}

const radarBoxes = Array.from({ length: 8 }, () => {
  const box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(5.1, 2.2, 2.5)), new THREE.LineBasicMaterial({ color: COLORS.radar }));
  box.visible = false;
  scene.add(box);
  return box;
});

function rayLines(max, color) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(max * 6), 3));
  geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(max * 6), 3));
  const lines = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85 }));
  lines.frustumCulled = false;
  lines.userData.color = new THREE.Color(color);
  scene.add(lines);
  return lines;
}
const lidarLines = rayLines(240, COLORS.lidar);
const sonarLines = rayLines(16, COLORS.sonar);
const gpsLine = rayLines(1, COLORS.gps);

function buildRoad(road) {
  scene.remove(roadGroup);
  roadGroup = new THREE.Group();
  const n = road.left.length;
  const pos = new Float32Array(n * 6);
  const idx = [];
  for (let i = 0; i < n; i++) {
    pos.set([road.left[i][0], 0.02, -road.left[i][1], road.right[i][0], 0.02, -road.right[i][1]], i * 6);
    if (i < n - 1) idx.push(2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  roadGroup.add(new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color: 0x3a3f46, side: THREE.DoubleSide })));

  const line = (pts, mat) => {
    const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts.map(([x, y]) => v3(x, y, 0.05))), mat);
    l.computeLineDistances();
    roadGroup.add(l);
  };
  const edge = new THREE.LineBasicMaterial({ color: 0xf2f2f2 });
  line(road.left, edge);
  line(road.right, edge);
  for (const lane of road.lanes) line(lane, new THREE.LineDashedMaterial({ color: 0xf2f2f2, dashSize: 3, gapSize: 6 }));

  // finish gate
  const [dx, dy, dh] = road.dest;
  const nx = -Math.sin(dh), ny = Math.cos(dh), half = road.width / 2 + 0.8;
  const post = new THREE.MeshLambertMaterial({ color: 0xffffff });
  for (const side of [-1, 1]) {
    const p = new THREE.Mesh(new THREE.BoxGeometry(0.3, 5, 0.3), post);
    p.position.copy(v3(dx + nx * half * side, dy + ny * half * side, 2.5));
    roadGroup.add(p);
  }
  const banner = new THREE.Mesh(new THREE.BoxGeometry(0.3, 1, half * 2), new THREE.MeshLambertMaterial({ color: COLORS.gps }));
  banner.position.copy(v3(dx, dy, 4.6));
  banner.rotation.y = dh;
  roadGroup.add(banner);
  hazardMeshes(road, roadGroup);
  scene.add(roadGroup);
}

// ---------- frames and interpolation ----------
let prev = null, curr = null, tCurr = 0, speed = 1, cfg = null;

function lerpAngle(a, b, t) {
  const d = Math.atan2(Math.sin(b - a), Math.cos(b - a));
  return a + d * t;
}

function poseAt(t) {
  const [x, y, yaw] = curr.ego;
  if (!prev) return { x, y, yaw };
  return { x: prev.ego[0] + (x - prev.ego[0]) * t, y: prev.ego[1] + (y - prev.ego[1]) * t, yaw: lerpAngle(prev.ego[2], yaw, t) };
}

function setRays(lines, data, maxRange, origin) {
  const pos = lines.geometry.attributes.position, col = lines.geometry.attributes.color;
  const n = data ? data[0].length : 0;
  const c = lines.userData.color;
  for (let i = 0; i < n; i++) {
    const a = data[0][i], d = data[1][i];
    const hit = d < maxRange - 0.01;
    const k = hit ? 1 : 0.18;
    pos.setXYZ(2 * i, origin.x, 1.0, -origin.y);
    pos.setXYZ(2 * i + 1, origin.x + Math.cos(a) * d, 1.0, -(origin.y + Math.sin(a) * d));
    col.setXYZ(2 * i, c.r * k, c.g * k, c.b * k);
    col.setXYZ(2 * i + 1, c.r * k, c.g * k, c.b * k);
  }
  lines.geometry.setDrawRange(0, n * 2);
  pos.needsUpdate = col.needsUpdate = true;
  lines.visible = n > 0;
}

function draw(t, dt) {
  if (!curr) return;
  const p = poseAt(t);
  ego.position.copy(v3(p.x, p.y));
  ego.rotation.y = p.yaw;

  const sameCars = prev && prev.traffic.length === curr.traffic.length;
  curr.traffic.forEach((car, i) => {
    if (!trafficMeshes[i]) scene.add((trafficMeshes[i] = carMesh(COLORS.traffic)));
    const m = trafficMeshes[i];
    const q = sameCars ? prev.traffic[i] : car;
    m.position.copy(v3(q[0] + (car[0] - q[0]) * t, q[1] + (car[1] - q[1]) * t));
    m.rotation.y = lerpAngle(q[2], car[2], t);
    m.children[0].material.color.setHex(car[3] ? COLORS.parked : COLORS.traffic);
    m.visible = true;
  });
  for (let i = curr.traffic.length; i < trafficMeshes.length; i++) trafficMeshes[i].visible = false;

  const peds = curr.pedestrians || [];
  peds.forEach((ped, i) => {
    if (!pedMeshes[i]) scene.add((pedMeshes[i] = pedMesh(i)));
    const q = prev?.pedestrians?.[i] || ped;
    pedMeshes[i].position.copy(v3(q[0] + (ped[0] - q[0]) * t, q[1] + (ped[1] - q[1]) * t));
    pedMeshes[i].rotation.y = ped[2];
    pedMeshes[i].visible = true;
  });
  for (let i = peds.length; i < pedMeshes.length; i++) pedMeshes[i].visible = false;
  for (const i of curr.holesHit || []) potholeMeshes[i]?.material.color.setHex(0x8a2b2b);  // potholes you hit turn red

  // rays are measured from the latest pose; shift them with the drawn car so they stay attached
  setRays(lidarLines, curr.lidar, cfg.lidar.range, p);
  setRays(sonarLines, curr.sonar, cfg.sonar.range, p);

  // radar targets: a car-sized box on cars, a person-sized one on pedestrians
  const tracked = curr.radar || [];
  radarBoxes.forEach((box, i) => {
    box.visible = i < tracked.length;
    if (!box.visible) return;
    const [x, y] = tracked[i];
    const car = curr.traffic.findIndex((c) => Math.hypot(c[0] - x, c[1] - y) < 0.5);
    if (car >= 0) {
      box.position.copy(trafficMeshes[car].position).setY(1.1);
      box.rotation.y = trafficMeshes[car].rotation.y;
      box.scale.set(1, 1, 1);
    } else {
      box.position.copy(v3(x, y, 0.95));
      box.scale.set(0.2, 0.9, 0.36);
    }
  });

  ghost.visible = !!curr.gps;
  if (curr.gps) {
    ghost.position.copy(v3(curr.gps[0], curr.gps[1], 0.05));
    ghost.rotation.y = p.yaw;
    setRays(gpsLine, [[Math.atan2(curr.gps[1] - p.y, curr.gps[0] - p.x)], [Math.hypot(curr.gps[0] - p.x, curr.gps[1] - p.y)]], Infinity, p);
  } else gpsLine.visible = false;

  updateCamera(p, dt);
}

// ---------- camera ----------
let camMode = "chase";
const camTarget = new THREE.Vector3();

// Smoothing per second, not per frame, so the camera keeps up on slow devices too.
const follow = (rate, dt) => 1 - Math.exp(-rate * dt);

function updateCamera(p, dt) {
  const fwd = new THREE.Vector3(Math.cos(p.yaw), 0, -Math.sin(p.yaw));
  const car = v3(p.x, p.y);
  if (camMode === "chase") {
    camera.up.set(0, 1, 0);
    camera.position.lerp(car.clone().addScaledVector(fwd, -13).setY(6), follow(6, dt));
    camTarget.lerp(car.clone().addScaledVector(fwd, 10).setY(1), follow(10, dt));
    camera.lookAt(camTarget);
  } else if (camMode === "top") {
    camera.position.copy(car).setY(55);
    camera.up.copy(fwd);
    camera.lookAt(car);
  } else {
    controls.target.copy(car);
    controls.update();
  }
}

function setCamera(mode) {
  camMode = mode;
  controls.enabled = mode === "free";
  if (mode === "free") {
    camera.up.set(0, 1, 0);
    const car = ego.position;
    camera.position.set(car.x - 30, 35, car.z + 30);
  }
  if (mode === "chase") camTarget.copy(ego.position);
}

// ---------- UI ----------
const worker = new Worker("worker.js");
let seed = Math.floor(Math.random() * 10000);
let scenario = null;
let endTimer = null;

function fmt(v, step) {
  return step >= 1 ? String(Math.round(v)) : Number(v).toFixed(step < 0.1 ? 2 : 1);
}

function slider(container, label, value, lo, hi, isInt, onInput, onChange) {
  // returns set(v): move the slider without firing its handlers
  const id = `s-${Math.random().toString(36).slice(2, 8)}`;
  const step = isInt ? 1 : (hi - lo) / 100;
  const wrap = document.createElement("div");
  wrap.className = "slider";
  wrap.innerHTML = `<label for="${id}">${label}</label><output for="${id}">${fmt(value, step)}</output>
    <input id="${id}" type="range" min="${lo}" max="${hi}" step="${step}" value="${value}">`;
  const input = wrap.querySelector("input"), out = wrap.querySelector("output");
  input.addEventListener("input", () => {
    out.textContent = fmt(input.value, step);
    onInput?.(Number(input.value));
  });
  if (onChange) input.addEventListener("change", () => onChange(Number(input.value)));
  container.appendChild(wrap);
  return (v) => {
    input.value = v;
    out.textContent = fmt(v, step);
  };
}

const scenarioSliders = {};

function buildControls({ defaults, ranges, scenario: sc, scenarioRanges, levels: lv }) {
  levels = lv;
  for (const [name, spec] of Object.entries(levels)) {
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.level = name;
    b.setAttribute("aria-pressed", "false");
    b.textContent = spec.title;
    b.addEventListener("click", () => startLevel(name));
    $("levels").appendChild(b);
  }
  document.querySelector('[data-level="free"]').addEventListener("click", () => startLevel(null));
  cfg = structuredClone(defaults);
  scenario = { ...sc };
  const box = $("sensors");
  const swatch = { lidar: COLORS.lidar, sonar: COLORS.sonar, radar: COLORS.radar, gps: COLORS.gps };
  for (const [name, values] of Object.entries(defaults)) {
    const el = document.createElement("div");
    el.className = "sensor" + (values.on ? "" : " off");
    const id = `t-${name}`;
    const colour = swatch[name] ? `<span class="swatch" style="background:#${swatch[name].toString(16).padStart(6, "0")}"></span>` : "";
    el.innerHTML = `<label class="toggle" for="${id}"><input id="${id}" type="checkbox" ${values.on ? "checked" : ""}>${LABELS[name]}${colour}</label><div class="sliders"></div>`;
    el.querySelector("input").addEventListener("change", (e) => setSensor(name, e.target.checked));
    const sliders = el.querySelector(".sliders");
    for (const [key, [lo, hi]] of Object.entries(ranges[name])) {
      slider(sliders, SETTING_LABELS[key] || key, values[key], lo, hi, key === "beams" || key === "tracks", (v) => {
        cfg[name][key] = v;
        worker.postMessage({ type: "config", sensors: { [name]: { [key]: v } } });
      });
    }
    box.appendChild(el);
  }
  for (const [key, [lo, hi]] of Object.entries(scenarioRanges)) {
    scenarioSliders[key] = slider($("scenario"), SCENARIO_LABELS[key] || key, sc[key], lo, hi, key === "lanes", null, (v) => {
      if (level) startLevel(null);  // levels are fixed; editing the road means free driving
      scenario[key] = v;
      reset();
    });
  }
}

function reset() {
  clearTimeout(endTimer);
  endTimer = null;
  $("banner").hidden = true;
  $("seed").value = seed;
  $("hud-seed").textContent = seed;
  prev = curr = null;
  worker.postMessage({ type: "reset", seed, scenario, horizon: level ? level.horizon : 1500 });
}

// ---------- levels: a fixed list of roads, scored as you go ----------
let levels = {}, level = null, levelIndex = 0, tally = null, freeScenario = null, levelDone = false;

function startLevel(name) {
  levelDone = false;
  pressGroup("data-level", document.querySelector(`[data-level="${name || "free"}"]`));
  if (!name) {
    if (level) scenario = { ...freeScenario };
    level = null;
    $("hud-level").hidden = true;
    $("level-note").textContent = "Any road you like. Pick a level for a fixed set of roads with pedestrians, debris and potholes, scored as you go.";
  } else {
    if (!level) freeScenario = { ...scenario };
    level = levels[name];
    scenario = { ...scenario, ...level.scenario };
    levelIndex = 0;
    tally = { success: 0, crash: 0, out_of_road: 0, hit_pedestrian: 0, damaged: 0, timeout: 0, potholes: 0 };
    seed = level.seeds[0];
    $("level-note").textContent = `${level.description} ${level.seeds.length} fixed roads; switching a sensor on or off restarts the level.`;
    $("hud-level").hidden = false;
    showTally();
  }
  for (const [k, set] of Object.entries(scenarioSliders)) set(scenario[k]);
  reset();
}

function showTally() {
  const n = level.seeds.length, done = Object.values(tally).reduce((a, b) => a + b, 0) - tally.potholes;
  const parts = [`<b>${level.title}</b> road ${Math.min(levelIndex + 1, n)}/${n}`, `${tally.success} arrived`];
  const extra = { hit_pedestrian: "pedestrians hit", crash: "crashes", damaged: "damaged", out_of_road: "off road", timeout: "out of time" };
  for (const [k, label] of Object.entries(extra)) if (tally[k]) parts.push(`${tally[k]} ${label}`);
  parts.push(`${tally.potholes} pothole${tally.potholes === 1 ? "" : "s"}`);
  $("hud-level").innerHTML = parts.join(" · ") + (done === n ? `<br><b>Score: ${Math.round((100 * tally.success) / n)}%</b>` : "");
}

function onFrame(frame) {
  prev = curr;
  curr = frame;
  tCurr = performance.now();
  $("hud-speed").textContent = Math.round(frame.ego[3] * 3.6);
  $("hud-progress").style.width = `${(frame.progress * 100).toFixed(1)}%`;
  $("hud-progress").parentElement.setAttribute("aria-valuenow", Math.round(frame.progress * 100));
  $("hud-status").textContent = frame.status === "driving" ? "Driving" : END_TEXT[frame.status];
  $("hud-holes").textContent = frame.potholes ? ` · ${frame.potholes} pothole${frame.potholes > 1 ? "s" : ""}` : "";
  if (frame.status !== "driving" && !endTimer && !levelDone) {
    const banner = $("banner");
    banner.textContent = END_TEXT[frame.status];
    banner.className = frame.status === "success" ? "ok" : "bad";
    banner.hidden = false;
    if (level) {
      tally[frame.status] += 1;
      tally.potholes += frame.potholes;
      showTally();
    }
    endTimer = setTimeout(() => {
      endTimer = null;
      if (!level) seed += 1;
      else if (++levelIndex < level.seeds.length) {
        seed = level.seeds[levelIndex];
        showTally();
      }
      else {  // level finished: leave the score up; pick the level again to replay
        levelDone = true;
        banner.textContent = `${level.title} done: ${tally.success} of ${level.seeds.length} roads`;
        banner.className = "ok";
        return;
      }
      reset();
    }, 2000);
  }
}

worker.onmessage = ({ data: msg }) => {
  if (msg.type === "status") $("loading-text").textContent = msg.text;
  else if (msg.type === "error") {
    $("loading").classList.add("error");
    $("loading-text").textContent = `Couldn't start the simulator: ${msg.text}`;
  } else if (msg.type === "settings") {
    buildControls(JSON.parse(msg.data));
    specialistsReady.then((list) => {
      specialists = list;
      buildPresets();
      updateDriver();
    });
    reset();
  } else if (msg.type === "driver") {
    if (!msg.ok) {
      $("driver-note").textContent = DRIVER_NOTES.missing;
      $("hud-driver").textContent = driverLabel("reference");
    }
  } else if (msg.type === "road") {
    const road = JSON.parse(msg.data);
    buildRoad(road);
    onFrame(road.frame);
    prev = null;
    setCamera(camMode);
    camera.position.copy(ego.position).add(new THREE.Vector3(-20, 10, 0));
    $("loading").classList.add("done");
  } else if (msg.type === "frame") onFrame(JSON.parse(msg.data));
};

$("new-road").addEventListener("click", () => {
  if (level) startLevel(null);
  seed = Math.floor(Math.random() * 100000);
  reset();
});
$("seed").addEventListener("change", (e) => {
  if (level) startLevel(null);
  seed = Math.max(0, Math.floor(Number(e.target.value) || 0));
  reset();
});

// ---------- driver: Auto picks the specialist trained on exactly the sensors that are on ----------
let driverMode = "auto", specialists = [], loadedModel = null;

const sensorKey = () => Object.keys(cfg).filter((n) => cfg[n].on).join("-");

function setSensor(name, on) {
  cfg[name].on = on;
  const box = $(`t-${name}`);
  box.checked = on;
  box.closest(".sensor").classList.toggle("off", !on);
  worker.postMessage({ type: "config", sensors: { [name]: { on } } });
  updateDriver();
  restartLevelSoon();
}

// A level score belongs to one sensor setup and driver, so changing either restarts it.
// Batched, so a preset that flips several sensors restarts it once.
let restartQueued = false;
function restartLevelSoon() {
  if (!level || restartQueued) return;
  restartQueued = true;
  queueMicrotask(() => {
    restartQueued = false;
    if (level) startLevel(Object.keys(levels).find((k) => levels[k] === level));
  });
}

function driverLabel(model) {
  if (model === "reference") return "Driver: reference lane follower";
  if (model === "generalist") return "Driver: generalist network";
  const sensors = model.replace("specialist-", "").split("-").map((n) => SHORT[n]).join(" + ");
  return `Driver: specialist (${sensors})`;
}

function updateDriver() {
  const key = sensorKey();
  const match = specialists.find((s) => s.sensors.join("-") === key);
  const pct = (x) => `${Math.round(x * 100)}%`;
  let model = driverMode, note = DRIVER_NOTES[driverMode];
  if (driverMode === "auto") {
    model = match ? match.model : "generalist";
    if (!match) note = DRIVER_NOTES.noSpecialist;
    else if (match.model === "generalist")
      note = `The specialist trained on these sensors finished ${pct(match.specialist)} of test roads, fewer than the generalist's ${pct(match.generalist)}, so the generalist is driving.`;
    else note = `Specialist trained on exactly these sensors. It finished ${pct(match.specialist)} of test roads (generalist: ${pct(match.generalist)}).`;
  }
  $("driver-note").textContent = note;
  document.querySelectorAll("#presets button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.key === key && !!match)));
  $("hud-driver").textContent = driverLabel(model);
  if (model !== loadedModel) {
    loadedModel = model;
    worker.postMessage({ type: "driver", name: model });
  }
}

function buildPresets() {
  const box = $("presets");
  for (const s of [...specialists].sort((a, b) => b.success - a.success)) {
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.key = s.sensors.join("-");
    b.innerHTML = `${s.sensors.map((n) => SHORT[n]).join(" + ")}<span>${Math.round(s.success * 100)}%</span>`;
    b.addEventListener("click", () => {
      for (const n of Object.keys(cfg)) if (cfg[n].on !== s.sensors.includes(n)) setSensor(n, s.sensors.includes(n));
      if (driverMode !== "auto") {
        driverMode = "auto";
        pressGroup("data-driver", document.querySelector('[data-driver="auto"]'));
        updateDriver();
      }
    });
    box.appendChild(b);
  }
}

document.querySelectorAll("[data-driver]").forEach((b) => b.addEventListener("click", () => {
  pressGroup("data-driver", b);
  driverMode = b.dataset.driver;
  updateDriver();
  restartLevelSoon();
}));

function pressGroup(attr, button) {
  document.querySelectorAll(`[${attr}]`).forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
}
document.querySelectorAll("[data-cam]").forEach((b) => b.addEventListener("click", () => {
  pressGroup("data-cam", b);
  setCamera(b.dataset.cam);
}));
document.querySelectorAll("[data-speed]").forEach((b) => b.addEventListener("click", () => {
  pressGroup("data-speed", b);
  speed = Number(b.dataset.speed);
  worker.postMessage({ type: "speed", value: speed });
}));
$("pause").addEventListener("click", (e) => {
  const on = e.currentTarget.getAttribute("aria-pressed") !== "true";
  e.currentTarget.setAttribute("aria-pressed", String(on));
  e.currentTarget.textContent = on ? "Resume" : "Pause";
  worker.postMessage({ type: "pause", value: on });
});
$("panel-toggle").addEventListener("click", (e) => {
  const open = $("panel").classList.toggle("open");
  e.currentTarget.setAttribute("aria-expanded", String(open));
  e.currentTarget.textContent = open ? "Close" : "Controls";
});

// ---------- render loop ----------
function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);
resize();

let lastRender = performance.now();
renderer.setAnimationLoop(() => {
  const now = performance.now();
  const dt = Math.min((now - lastRender) / 1000, 1);
  lastRender = now;
  draw(Math.min((now - tCurr) / (STEP_MS / speed), 1), dt);
  renderer.render(scene, camera);
});
