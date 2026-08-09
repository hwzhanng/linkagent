import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import URDFLoader from "urdf-loader";

const $ = (selector) => document.querySelector(selector);
const pageParameters = new URLSearchParams(window.location.search);
const calibratedV2 = pageParameters.get("mapping") === "v2";
const mappingEndpoint = calibratedV2 ? "/api/map-v2" : "/api/map";
if (pageParameters.get("capture") === "1") document.body.classList.add("capture-mode");

function requestedCaptureDirection() {
  const directions = {
    palm_front: new THREE.Vector3(0, 1, 0.18),
    palm_back: new THREE.Vector3(0, -1, 0.18),
    palmar_side_front: new THREE.Vector3(0.72, 1, 0.18),
    palmar_side_back: new THREE.Vector3(0.72, -1, 0.18),
    side_left: new THREE.Vector3(1, 0, 0.15),
    side_right: new THREE.Vector3(-1, 0, 0.15),
  };
  return directions[pageParameters.get("camera")] ?? directions.palmar_side_front;
}
const viewport = $("#viewport");
const loadingOverlay = $("#loading-overlay");
const statusDot = $("#connection-dot");
const statusTitle = $("#status-title");
const statusDetail = $("#status-detail");
const saveResult = $("#save-result");

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 0.001, 10);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.075;
controls.minDistance = 0.12;
controls.maxDistance = 1.4;

scene.add(new THREE.HemisphereLight(0xd9fff0, 0x18201d, 2.1));
const keyLight = new THREE.DirectionalLight(0xffffff, 3.2);
keyLight.position.set(0.45, -0.35, 0.55);
keyLight.castShadow = true;
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x64dba0, 2.0);
rimLight.position.set(-0.35, 0.25, 0.15);
scene.add(rimLight);

const axes = new THREE.AxesHelper(0.08);
axes.visible = false;
scene.add(axes);

let config = null;
let robot = null;
let bounds = null;
let baseTargets = {};
let allTargets = {};
let synergy = [0, 0, 0, 0, 0];
let overrides = new Map();
const synergyRows = [];
const jointRows = new Map();
const thumbRootRows = new Map();
const poseStates = new Map();
let currentPoseKey = "center";
let requestSerial = 0;
let pendingMapFrame = 0;

const poseAnchors = [
  { key: "center", label: "中心预抓取", values: [0, 0, 0, 0, 0] },
  { key: "flex_open", label: "1− 整体张开", values: [-1, 0, 0, 0, 0] },
  { key: "flex_close", label: "1+ 整体闭合", values: [1, 0, 0, 0, 0] },
  { key: "curl_proximal", label: "2− 近端卷曲", values: [0, -1, 0, 0, 0] },
  { key: "curl_distal", label: "2+ 远端卷曲", values: [0, 1, 0, 0, 0] },
  { key: "balance_ulnar", label: "3− 尺侧加强", values: [0, 0, -1, 0, 0] },
  { key: "balance_radial", label: "3+ 桡侧加强", values: [0, 0, 1, 0, 0] },
  { key: "spread_negative", label: "4− 展收负端", values: [0, 0, 0, -1, 0] },
  { key: "spread_positive", label: "4+ 展收正端", values: [0, 0, 0, 1, 0] },
  { key: "thumb_open", label: "5− 拇指打开", values: [0, 0, 0, 0, -1] },
  { key: "thumb_oppose", label: "5+ 拇指对掌", values: [0, 0, 0, 0, 1] },
];

const thumbRootJoints = [
  { name: "right_hand_thumb_joint0", label: "根部侧摆" },
  { name: "right_hand_thumb_joint1", label: "根部滚转" },
];

const synergyCopy = {
  global_finger_flexion: ["整体屈曲", "四指共同开合"],
  proximal_distal_curl_shape: ["卷曲形状", "近端与远端的屈曲分配"],
  radial_ulnar_flexion_balance: ["桡尺侧平衡", "食指侧与小指侧的闭合平衡"],
  finger_spread: ["四指展收", "四指根部侧向展开"],
  thumb_opposition: ["拇指对掌", "拇指根部与末端协同屈曲"],
};

function setStatus(kind, title, detail) {
  statusDot.className = `status-dot ${kind}`;
  statusTitle.textContent = title;
  statusDetail.textContent = detail;
}

function resizeRenderer() {
  const { clientWidth, clientHeight } = viewport;
  renderer.setSize(clientWidth, clientHeight, false);
  camera.aspect = clientWidth / Math.max(clientHeight, 1);
  camera.updateProjectionMatrix();
}

function fitCamera(direction = new THREE.Vector3(0.15, 1.0, 0.32)) {
  if (!robot) return;
  bounds = new THREE.Box3().setFromObject(robot);
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z) * 0.68;
  controls.target.copy(center);
  camera.position.copy(center).add(direction.normalize().multiplyScalar(Math.max(radius * 2.5, 0.18)));
  camera.near = Math.max(radius / 100, 0.0005);
  camera.far = Math.max(radius * 25, 2);
  camera.updateProjectionMatrix();
  controls.update();
}

async function fitCameraToCalibratedEnvelope(
  direction = new THREE.Vector3(0.15, 1.0, 0.32),
) {
  if (!robot || !calibratedV2) return fitCamera(direction);
  const requestedTargets = { ...baseTargets };
  const envelope = new THREE.Box3();
  const anchors = [[0, 0, 0, 0, 0]];
  for (let coordinate = 0; coordinate < 5; coordinate += 1) {
    for (const sign of [-1, 1]) {
      const values = [0, 0, 0, 0, 0];
      values[coordinate] = sign;
      anchors.push(values);
    }
  }
  for (const values of anchors) {
    const response = await fetch(
      `${mappingEndpoint}?${new URLSearchParams({ values: values.join(",") })}`,
    );
    if (!response.ok) throw new Error(`v2联合包围盒映射失败：${response.status}`);
    baseTargets = (await response.json()).active_joint_targets;
    applyJointTargets();
    robot.updateMatrixWorld(true);
    envelope.union(new THREE.Box3().setFromObject(robot));
  }
  baseTargets = requestedTargets;
  applyJointTargets();
  robot.updateMatrixWorld(true);
  bounds = envelope;
  const center = envelope.getCenter(new THREE.Vector3());
  const size = envelope.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z) * 0.68;
  controls.target.copy(center);
  camera.position.copy(center).add(
    direction.normalize().multiplyScalar(Math.max(radius * 2.5, 0.18)),
  );
  camera.near = Math.max(radius / 100, 0.0005);
  camera.far = Math.max(radius * 25, 2);
  camera.updateProjectionMatrix();
  controls.update();
}

function applyJointTargets() {
  if (!robot || !config) return;
  allTargets = { ...baseTargets };
  for (const [name, value] of overrides) allTargets[name] = value;
  for (const mimic of config.mimic_joints) {
    const driver = allTargets[mimic.driver];
    if (Number.isFinite(driver)) {
      allTargets[mimic.name] = THREE.MathUtils.clamp(
        mimic.multiplier * driver + mimic.offset,
        mimic.lower,
        mimic.upper,
      );
    }
  }
  for (const [name, value] of Object.entries(allTargets)) {
    const joint = robot.joints[name];
    if (joint && Number.isFinite(value)) joint.setJointValue(value);
  }
  refreshJointRows();
  refreshThumbRootRows();
  const found = config.active_joints.filter((joint) => Boolean(robot.joints[joint.name])).length;
  $("#joint-coverage").textContent = `${found}/16`;
}

function createThumbRootControls() {
  const host = $("#thumb-root-controls");
  host.replaceChildren();
  for (const definition of thumbRootJoints) {
    const joint = config.active_joints.find((item) => item.name === definition.name);
    if (!joint) throw new Error(`缺少拇指根关节：${definition.name}`);
    const row = document.createElement("div");
    row.className = "thumb-root-row";
    row.innerHTML = `
      <div class="thumb-root-meta">
        <span class="thumb-root-name"><strong>${definition.label}</strong><code>${definition.name.replace("right_hand_", "")}</code></span>
        <span class="thumb-root-actions">
          <output class="thumb-root-value">—</output>
          <span class="thumb-root-mode">跟随5D</span>
          <button class="thumb-root-reset" type="button">撤销</button>
        </span>
      </div>
      <input type="range" min="${joint.lower}" max="${joint.upper}" step="0.001" value="${joint.neutral}" aria-label="${definition.label}" />
    `;
    const input = row.querySelector("input");
    const output = row.querySelector("output");
    const mode = row.querySelector(".thumb-root-mode");
    input.addEventListener("input", () => {
      overrides.set(joint.name, Number(input.value));
      rememberCurrentPose();
      applyJointTargets();
    });
    row.querySelector(".thumb-root-reset").addEventListener("click", () => {
      overrides.delete(joint.name);
      rememberCurrentPose();
      applyJointTargets();
    });
    thumbRootRows.set(joint.name, { row, input, output, mode });
    host.appendChild(row);
  }
}

function refreshThumbRootRows() {
  if (!config) return;
  for (const definition of thumbRootJoints) {
    const controlsForJoint = thumbRootRows.get(definition.name);
    if (!controlsForJoint) continue;
    const value = allTargets[definition.name];
    if (!Number.isFinite(value)) continue;
    const overridden = overrides.has(definition.name);
    controlsForJoint.input.value = String(value);
    controlsForJoint.output.textContent = `${value.toFixed(3)} / ${THREE.MathUtils.radToDeg(value).toFixed(1)}°`;
    controlsForJoint.mode.textContent = overridden ? "独立覆盖" : "跟随5D";
    controlsForJoint.row.classList.toggle("overridden", overridden);
  }
}

function requestMapping() {
  cancelAnimationFrame(pendingMapFrame);
  pendingMapFrame = requestAnimationFrame(async () => {
    const serial = ++requestSerial;
    const params = new URLSearchParams({ values: synergy.join(",") });
    try {
      const response = await fetch(`${mappingEndpoint}?${params}`);
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      if (serial !== requestSerial) return;
      baseTargets = payload.active_joint_targets;
      applyJointTargets();
      setStatus("ready", "手部模型已就绪", "5D 映射来自当前 Python 实现");
    } catch (error) {
      setStatus("error", "5D 映射失败", String(error));
    }
  });
}

function rememberCurrentPose() {
  poseStates.set(currentPoseKey, {
    synergy: [...synergy],
    overrides: new Map(overrides),
  });
}

function syncSynergyRows() {
  synergyRows.forEach(({ input, output }, index) => {
    input.value = String(synergy[index]);
    output.textContent = synergy[index].toFixed(2);
  });
}

function selectPose(key) {
  rememberCurrentPose();
  currentPoseKey = key;
  const state = poseStates.get(key);
  if (!state) return;
  synergy = [...state.synergy];
  overrides = new Map(state.overrides);
  syncSynergyRows();
  $("#pose-anchor").value = key;
  $("#pose-status").textContent = `${poseAnchors.find((item) => item.key === key)?.label ?? key} · 调整独立保存于当前页面`;
  requestMapping();
}

function createPoseControls() {
  const select = $("#pose-anchor");
  select.replaceChildren();
  for (const anchor of poseAnchors) {
    const option = document.createElement("option");
    option.value = anchor.key;
    option.textContent = anchor.label;
    select.appendChild(option);
  }
  select.value = currentPoseKey;
  select.addEventListener("change", () => selectPose(select.value));
}

function restoreDraftPoseStates(payload, requestedPose = null) {
  if (payload.schema_version !== 1 || !payload.pose_drafts) {
    throw new Error("上次草案格式不兼容");
  }
  for (const anchor of poseAnchors) {
    const saved = payload.pose_drafts[anchor.key];
    if (!saved || !Array.isArray(saved.synergy) || saved.synergy.length !== 5) {
      throw new Error(`上次草案缺少姿态：${anchor.key}`);
    }
    poseStates.set(anchor.key, {
      synergy: saved.synergy.map(Number),
      overrides: new Map(
        Object.entries(saved.per_joint_overrides_rad ?? {}).map(
          ([name, value]) => [name, Number(value)],
        ),
      ),
    });
  }
  currentPoseKey = requestedPose && poseStates.has(requestedPose)
    ? requestedPose
    : (poseStates.has(payload.selected_pose) ? payload.selected_pose : "center");
  const state = poseStates.get(currentPoseKey);
  synergy = [...state.synergy];
  overrides = new Map(state.overrides);
}

async function fetchLatestDraft() {
  const response = await fetch("/api/latest-draft", { cache: "no-store" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error ?? "上次草案加载失败");
  return payload;
}

async function loadLatestDraft(requestedPose = null) {
  const payload = await fetchLatestDraft();
  restoreDraftPoseStates(payload, requestedPose);
  $("#pose-anchor").value = currentPoseKey;
  syncSynergyRows();
  requestMapping();
  const label = poseAnchors.find((item) => item.key === currentPoseKey)?.label ?? currentPoseKey;
  $("#pose-status").textContent = `${label} · 已恢复上次保存`;
  showSaveResult(`已加载：${payload.created_at ?? "上次保存"}`);
}

function createSynergyControls() {
  const host = $("#synergy-controls");
  host.replaceChildren();
  config.synergy_order.forEach((name, index) => {
    const [label, description] = synergyCopy[name] ?? [name, "归一化策略坐标"];
    const row = document.createElement("label");
    row.className = "slider-row";
    row.innerHTML = `
      <div class="slider-meta">
        <span class="slider-name"><strong>${index + 1}. ${label}</strong><small>${description}</small></span>
        <output class="slider-value">${synergy[index].toFixed(2)}</output>
      </div>
      <input type="range" min="-1" max="1" step="0.01" value="${synergy[index]}" aria-label="${label}" />
    `;
    const input = row.querySelector("input");
    const output = row.querySelector("output");
    input.addEventListener("input", () => {
      synergy[index] = Number(input.value);
      output.textContent = synergy[index].toFixed(2);
      rememberCurrentPose();
      requestMapping();
    });
    synergyRows.push({ input, output });
    host.appendChild(row);
  });
}

function createJointControls() {
  const host = $("#joint-controls");
  host.replaceChildren();
  config.active_joints.forEach((joint) => {
    const row = document.createElement("div");
    row.className = "joint-row";
    row.innerHTML = `
      <div class="joint-header">
        <code>${joint.name.replace("right_hand_", "")}</code>
        <span><span class="joint-reading">—</span><button class="joint-reset" type="button">撤销</button></span>
      </div>
      <input type="range" min="${joint.lower}" max="${joint.upper}" step="0.001" value="${joint.neutral}" aria-label="${joint.name}" />
    `;
    const input = row.querySelector("input");
    const reading = row.querySelector(".joint-reading");
    const reset = row.querySelector(".joint-reset");
    input.addEventListener("input", () => {
      overrides.set(joint.name, Number(input.value));
      rememberCurrentPose();
      applyJointTargets();
    });
    reset.addEventListener("click", () => {
      overrides.delete(joint.name);
      rememberCurrentPose();
      applyJointTargets();
    });
    jointRows.set(joint.name, { row, input, reading });
    host.appendChild(row);
  });
}

function refreshJointRows() {
  if (!config) return;
  for (const joint of config.active_joints) {
    const controlsForJoint = jointRows.get(joint.name);
    if (!controlsForJoint) continue;
    const value = allTargets[joint.name] ?? joint.neutral;
    controlsForJoint.input.value = String(value);
    controlsForJoint.reading.textContent = `${value.toFixed(3)} / ${THREE.MathUtils.radToDeg(value).toFixed(1)}°`;
    controlsForJoint.row.classList.toggle("overridden", overrides.has(joint.name));
  }
  $("#override-count").textContent = `${overrides.size} 覆盖`;
}

function resetSynergy() {
  const anchor = poseAnchors.find((item) => item.key === currentPoseKey);
  synergy = currentPoseKey === "center"
    ? [...config.pregrasp_synergy]
    : [...(anchor?.values ?? config.pregrasp_synergy)];
  overrides.clear();
  rememberCurrentPose();
  syncSynergyRows();
  requestMapping();
}

function draftPayload() {
  rememberCurrentPose();
  return {
    schema_version: 1,
    created_at: new Date().toISOString(),
    purpose: "tomato_pick_g20_5d_hand_manifold_draft",
    source: config.source,
    synergy_order: config.synergy_order,
    synergy_values: Object.fromEntries(config.synergy_order.map((name, index) => [name, synergy[index]])),
    active_joint_targets_rad: Object.fromEntries(config.active_joints.map((joint) => [joint.name, allTargets[joint.name]])),
    active_joint_targets_deg: Object.fromEntries(config.active_joints.map((joint) => [joint.name, THREE.MathUtils.radToDeg(allTargets[joint.name])])),
    mimic_joint_targets_rad: Object.fromEntries(config.mimic_joints.map((joint) => [joint.name, allTargets[joint.name]])),
    per_joint_overrides_rad: Object.fromEntries(overrides),
    selected_pose: currentPoseKey,
    pose_drafts: Object.fromEntries(poseAnchors.map((anchor) => {
      const state = poseStates.get(anchor.key);
      return [anchor.key, {
        label: anchor.label,
        synergy: state?.synergy ?? anchor.values,
        per_joint_overrides_rad: Object.fromEntries(state?.overrides ?? []),
      }];
    })),
    note: "Draft only; does not modify production training or simulation configuration.",
  };
}

async function copyDraft() {
  await navigator.clipboard.writeText(JSON.stringify(draftPayload(), null, 2));
  showSaveResult("已复制到剪贴板");
}

function downloadDraft() {
  const blob = new Blob([JSON.stringify(draftPayload(), null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `tomato_hand_5d_draft_${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
  showSaveResult("草案已下载");
}

async function saveDraft() {
  const response = await fetch("/api/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(draftPayload()),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error ?? "保存失败");
  showSaveResult(`已保存：${payload.path}`);
}

function showSaveResult(message, isError = false) {
  saveResult.textContent = message;
  saveResult.classList.toggle("error", isError);
}

function wireToolbar() {
  $("#fit-view").addEventListener("click", () => fitCamera());
  $("#palm-view").addEventListener("click", () => fitCamera(new THREE.Vector3(0, 1, 0.18)));
  $("#side-view").addEventListener("click", () => fitCamera(new THREE.Vector3(1, 0, 0.15)));
  $("#auto-rotate").addEventListener("change", (event) => { controls.autoRotate = event.target.checked; controls.autoRotateSpeed = 1.4; });
  $("#show-axes").addEventListener("change", (event) => { axes.visible = event.target.checked; });
  $("#reset-synergy").addEventListener("click", resetSynergy);
  $("#load-latest").addEventListener("click", () => loadLatestDraft().catch((error) => showSaveResult(String(error), true)));
  $("#clear-overrides").addEventListener("click", () => { overrides.clear(); rememberCurrentPose(); applyJointTargets(); });
  $("#copy-json").addEventListener("click", () => copyDraft().catch((error) => showSaveResult(String(error), true)));
  $("#download-json").addEventListener("click", downloadDraft);
  $("#save-draft").addEventListener("click", () => saveDraft().catch((error) => showSaveResult(String(error), true)));
}

async function loadViewer() {
  const configResponse = await fetch("/api/config.json");
  if (!configResponse.ok) throw new Error(`配置加载失败：${configResponse.status}`);
  config = await configResponse.json();
  synergy = [...config.pregrasp_synergy];
  for (const anchor of poseAnchors) {
    poseStates.set(anchor.key, {
      synergy: anchor.key === "center" ? [...config.pregrasp_synergy] : [...anchor.values],
      overrides: new Map(),
    });
  }
  const searchParams = pageParameters;
  const requestedPose = searchParams.get("pose");
  if (searchParams.get("load") === "latest") {
    restoreDraftPoseStates(await fetchLatestDraft(), requestedPose);
  } else {
    if (requestedPose && poseStates.has(requestedPose)) currentPoseKey = requestedPose;
    synergy = [...poseStates.get(currentPoseKey).synergy];
  }
  const requestedValues = searchParams.get("values")
    ?.split(",")
    .map(Number);
  if (
    requestedValues?.length === 5
    && requestedValues.every((value) => Number.isFinite(value) && value >= -1 && value <= 1)
  ) {
    synergy = requestedValues;
  }
  createPoseControls();
  createSynergyControls();
  createJointControls();
  createThumbRootControls();
  wireToolbar();
  const initialMapResponse = await fetch(`${mappingEndpoint}?${new URLSearchParams({ values: synergy.join(",") })}`);
  if (!initialMapResponse.ok) throw new Error(`初始5D映射失败：${initialMapResponse.status}`);
  baseTargets = (await initialMapResponse.json()).active_joint_targets;
  await new Promise((resolve, reject) => {
    const manager = new THREE.LoadingManager();
    const loader = new URDFLoader(manager);
    let parsedRobot = false;
    let finished = false;

    manager.onError = (url) => reject(new Error(`网格加载失败：${url}`));
    manager.onLoad = () => {
      if (!parsedRobot || finished) return;
      finished = true;
      requestAnimationFrame(() => requestAnimationFrame(async () => {
        applyJointTargets();
        if (document.body.classList.contains("capture-mode") && calibratedV2) {
          await fitCameraToCalibratedEnvelope(requestedCaptureDirection());
        } else {
          fitCamera();
        }
        setStatus(
          "ready",
          "手部模型已就绪",
          calibratedV2 ? "用户标定 v2 连续流形" : "Clean-v5 旧版 5D 映射",
        );
        loadingOverlay.classList.add("hidden");
        loadingOverlay.style.display = "none";
        window.__HAND_5D_VIEWER__ = {
          ready: true,
          jointCount: Object.keys(robot.joints).length,
          linkCount: Object.keys(robot.links).length,
          source: config.source,
          mapping: calibratedV2 ? "tomato_g20_user_calibrated_5d_v2" : "clean_v5_legacy_5d",
          getDraft: draftPayload,
        };
        resolve();
      }));
    };
    loader.parseCollision = false;
    loader.load(
      "/model/right_hand_only.urdf",
      (loadedRobot) => {
        robot = loadedRobot;
        robot.name = "tomato_pick_right_hand_only";
        robot.rotation.set(-Math.PI / 2, 0, Math.PI);
        robot.traverse((object) => {
          if (!object.isMesh) return;
          object.castShadow = true;
          object.receiveShadow = true;
          const material = new THREE.MeshStandardMaterial({
            color: object.material?.color ?? new THREE.Color(0xd8e3dc),
            roughness: 0.44,
            metalness: 0.12,
          });
          object.material = material;
        });
        scene.add(robot);
        parsedRobot = true;
      },
      undefined,
      reject,
    );
  });
}

function animate() {
  requestAnimationFrame(animate);
  resizeRenderer();
  controls.update();
  renderer.render(scene, camera);
}

animate();
loadViewer().catch((error) => {
  console.error(error);
  setStatus("error", "手部资产加载失败", String(error));
  loadingOverlay.querySelector("span").textContent = `加载失败：${error}`;
  loadingOverlay.querySelector(".spinner").style.display = "none";
  window.__HAND_5D_VIEWER__ = { ready: false, error: String(error) };
});
