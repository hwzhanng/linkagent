import * as THREE from 'three';
import { OrbitControls } from '/vendor/OrbitControls.js';
import URDFLoader from '/vendor/URDFLoader.js';
import { g20LogicalToUrdf } from '/hand-kinematics.js';

const $ = (id) => document.getElementById(id);
let state = null, plan = null, token = '', previewTimer = null, robot = null;
let toastTimer = null;

async function api(path, payload) {
  const options = payload === undefined ? {} : { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) };
  const response = await fetch(path, options); const data = await response.json().catch(() => ({}));
  if (!response.ok || data.ok === false) throw new Error(data.error || `请求失败 ${response.status}`);
  return data;
}
function toast(message, bad = false) { const node = $('toast'); node.textContent = message; node.classList.toggle('bad', bad); node.classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => node.classList.remove('show'), 2800); }
function clamp(v, a, b) { return Math.max(a, Math.min(b, Number(v))); }
function configFromUi() { return { diameter_mm: Number($('diameter').value), height_mm: Number($('height').value), palm_support: $('palmSupport').checked }; }
const outputs = { diameter: 'diameterOut', height: 'heightOut' };
Object.entries(outputs).forEach(([id, out]) => $(id).addEventListener('input', () => { $(out).textContent = $(id).value; }));
document.querySelectorAll('.tab').forEach((button) => button.addEventListener('click', () => { document.querySelectorAll('.tab,.tab-panel').forEach((node) => node.classList.remove('active')); button.classList.add('active'); document.querySelector(`[data-panel="${button.dataset.tab}"]`).classList.add('active'); }));

// URDF viewer --------------------------------------------------------------
const host = $('viewport'); const scene = new THREE.Scene(); const camera = new THREE.PerspectiveCamera(30, 1, .001, 10); const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' }); renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2)); renderer.outputColorSpace = THREE.SRGBColorSpace; host.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.enablePan = false; scene.add(new THREE.HemisphereLight(0xc9e5ff, 0x152333, 1.8)); const key = new THREE.DirectionalLight(0xffffff, 3); key.position.set(1.5, -2, 3); scene.add(key); const fill = new THREE.DirectionalLight(0xffa56b, 1.5); fill.position.set(-2, 1, 1); scene.add(fill);
let radius = .18;
function resize() { const w = Math.max(1, host.clientWidth), h = Math.max(1, host.clientHeight); renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
new ResizeObserver(resize).observe(host); resize();
// 模型现在放大为旧版本的两倍；缩短相机距离，确保画面中的手也同步放大。
function fitView() { const d = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov) / 2) * 1.42; camera.position.set(d * .55, d * .75, -d * .65); controls.target.set(0, 0, 0); controls.minDistance = radius * 1.15; controls.maxDistance = radius * 18; controls.update(); }
function applyPose(pose) { if (!robot || !Array.isArray(pose)) return; const joints = g20LogicalToUrdf(pose); Object.entries(joints || {}).forEach(([name, angle]) => { if (robot.joints?.[name]) robot.setJointValue(name, angle); }); robot.updateMatrixWorld(true); }
async function loadRobot() {
  try {
    const base = '/hand-model/L20_8_l_s';
    const cfgResponse = await fetch(`${base}/model.json`);
    if (!cfgResponse.ok) throw new Error(`模型配置加载失败 HTTP ${cfgResponse.status}`);
    const cfg = await cfgResponse.json();
    const manager = new THREE.LoadingManager();
    let managerError = '';
    manager.onError = (url) => { managerError = `网格资源加载失败：${url}`; };
    const loader = new URDFLoader(manager);
    loader.packages = {[cfg.package_name]: base};
    loader.workingPath = `${base}/`;
    loader.parseCollision = false;
    let loaded = null;
    const ready = new Promise((resolve, reject) => {
      manager.onLoad = () => managerError ? reject(new Error(managerError)) : resolve();
      loader.load(`${base}/${cfg.urdf}`, (model) => { loaded = model; }, undefined, reject);
    });
    await ready;
    robot = loaded;
    if (!robot) throw new Error('URDFLoader 未返回模型');
    // 左手资产的厂商坐标需要绕 X/Z 变换到 Three.js 的竖直视图。
    // 之前额外叠加 Y=π 会把手掌上下翻转；恢复已验证的左手基准姿态。
    // 在已校正的左手姿态基础上绕 Z 轴顺时针 90° 展示。
    robot.rotation.set(-Math.PI / 2, 0, Math.PI / 2);
    robot.scale.setScalar(1.0);
    scene.add(robot);
    robot.traverse((node) => { if (node.isMesh) node.material = new THREE.MeshStandardMaterial({color:0xb8c4cc, roughness:.5, metalness:.15}); });
    const box = new THREE.Box3().setFromObject(robot);
    robot.position.sub(box.getCenter(new THREE.Vector3()));
    robot.updateMatrixWorld(true);
    radius = Math.max(.045, box.getBoundingSphere(new THREE.Sphere()).radius);
    // 世界参考坐标轴固定在模型复位后的原点：X 红、Y 绿、Z 蓝。
    // 轴长度略大于掌部半径，便于检查上下方向和关节运动方向。
    const axes = new THREE.AxesHelper(Math.max(0.22, radius * 0.9));
    axes.name = 'tomato-hand-reference-axes';
    scene.add(axes);
    fitView();
    $('loading').style.display = 'none';
  } catch (error) { $('loading').textContent = `URDF 加载失败：${error.message}`; }
}
function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); } animate(); $('resetView').addEventListener('click', fitView); loadRobot();

function renderPlan(data) {
  plan = data.plan; token = data.token; $('planEmpty').classList.add('hidden'); $('planResult').classList.remove('hidden'); $('strategyLabel').textContent = '软网球三指基线 · 几何触觉自适应'; $('planId').textContent = `plan ${plan.plan_id}`; $('thumbTarget').textContent = `${plan.thumb_alignment.soft_target} ± ${plan.thumb_alignment.tolerance} raw`; $('referenceWeight').textContent = `${Math.round(plan.reference_weight * 100)}%（弱参考）`; $('referenceTolerance').textContent = `${plan.reference_envelope.frame_tolerance} / ${plan.reference_envelope.grasp_tolerance} raw`; $('palmResult').textContent = plan.palm_support_required ? '要求开启' : '未开启'; $('fingerPlan').innerHTML = plan.fingers.map((f) => `<div><b>${f.pressure_target}</b>${f.name}<small>独立目标</small></div>`).join(''); $('stagePlan').innerHTML = plan.stages.map((stage) => `<li><span>${stage.label}</span> · ${stage.duration_ms ? `${stage.duration_ms} ms` : '持续'} · 步长 ≤ ${stage.max_step}</li>`).join(''); document.querySelectorAll('[data-check]').forEach((row) => { const ok = plan.checks[row.dataset.check] === true; row.classList.toggle('pass', ok); row.classList.toggle('fail', !ok); row.querySelector('i').textContent = ok ? '✓' : '!'; }); $('previewButton').disabled = false; $('previewMessage').textContent = '三指方案已生成。请在下方 URDF 视图确认几何净空。'; toast('三指自适应包络方案已生成'); }
$('planButton').addEventListener('click', async () => { try { renderPlan(await api('/api/tomato/config', configFromUi())); } catch (error) { toast(error.message, true); } });

function interpolate(a, b, t) { const e = t * t * (3 - 2 * t); return a.map((v, i) => Math.round(v + (b[i] - v) * e)); }
function playPreview() { if (!plan || !robot) return; if (previewTimer) cancelAnimationFrame(previewTimer); const frames = plan.stages.filter((x) => x.duration_ms > 0); let index = 0, from = state?.pose || plan.stages[0].pose, started = performance.now(); $('previewMessage').textContent = '正在播放候选轨迹，仅更新 URDF…'; function tick(now) { if (index >= frames.length) { applyPose(frames.at(-1).pose); $('previewMessage').textContent = '预演完成。请确认净空后勾选现场确认。'; $('startButton').disabled = !$('safetyConfirm').checked; previewTimer = null; return; } const frame = frames[index], t = Math.min(1, (now - started) / frame.duration_ms); applyPose(interpolate(from, frame.pose, t)); if (t >= 1) { from = frame.pose; index += 1; started = now; } previewTimer = requestAnimationFrame(tick); } previewTimer = requestAnimationFrame(tick); }
$('previewButton').addEventListener('click', () => { $('viewport').scrollIntoView({behavior:'smooth', block:'center'}); playPreview(); });
$('safetyConfirm').addEventListener('change', () => { $('startButton').disabled = !$('safetyConfirm').checked || !token; });
$('startButton').addEventListener('click', async () => { if (!window.confirm('确认已完成预演，允许按自适应触觉策略低速抓取番茄？')) return; try { await api('/api/tomato/start', {token, confirm:'TOMATO_GRASP_CONFIRM'}); toast('已开始自适应抓取'); } catch (error) { if (/预演令牌|预演计划/.test(error.message)) { token = ''; $('startButton').disabled = true; $('previewMessage').textContent = '当前预演已失效，请重新生成方案并播放一次 URDF 预演。'; } toast(error.message, true); } });

function heat(v) { const t = clamp(v, 0, 255) / 255; return `rgb(${Math.round(20 + 220 * t)},${Math.round(70 + 120 * (1 - Math.abs(t - .55) * 1.5))},${Math.round(100 - 70 * t)})`; }
function buildMatrices() { $('matrices').innerHTML = ['拇指','食指','中指'].map((name, i) => `<div class="matrix"><h3>${name}</h3><canvas width="120" height="170"></canvas><small>峰值 <b id="peak${i}">0</b> · 载荷 <b id="load${i}">0</b></small></div>`).join(''); $('contactSummary').innerHTML = ['拇指','食指','中指'].map((name) => `<span>${name} · 无接触</span>`).join(''); }
function drawMatrices(s) { const mats = s.matrices || []; const contacts = s.contacts || s.raw_contacts || []; document.querySelectorAll('.matrix').forEach((card, i) => { const ctx = card.querySelector('canvas').getContext('2d'), matrix = mats[i] || []; for (let r = 0; r < 12; r += 1) for (let c = 0; c < 6; c += 1) { ctx.fillStyle = heat(matrix?.[r]?.[c] || 0); ctx.fillRect(c * 20 + 1, r * 14 + 1, 18, 12); } $(`peak${i}`).textContent = Math.round(s.peaks?.[i] || 0); $(`load${i}`).textContent = Math.round(s.loads?.[i] || 0); const pill = $('contactSummary').children[i], on = Boolean(contacts[i]); pill.classList.toggle('on', on); pill.textContent = `${['拇指','食指','中指'][i]} · ${on ? '接触' : '无接触'}`; }); }
function updateState(s) { state = s; if (token && !s.preview_valid) { token = ''; $('startButton').disabled = true; if (!previewTimer) $('previewMessage').textContent = '服务已重启或预演已过期，请重新生成方案并播放一次 URDF 预演。'; } $('connectionBadge').className = `status ${s.connected ? 'online' : 'offline'}`; $('connectionBadge').querySelector('span').textContent = s.source === 'hardware' ? '左手已连接' : '仿真服务'; $('connectionDetail').textContent = s.motion_enabled ? (s.actual_armed ? '实机已解锁' : '运动权限已开 · 未解锁') : '只读/仿真，不发送命令'; $('hwConnected').textContent = s.connected ? '在线' : '离线'; $('hwMotion').textContent = s.motion_enabled ? '已授权' : '未授权'; $('hwArmed').textContent = s.actual_armed ? '已解锁' : '未解锁'; $('phase').textContent = s.phase || '待机'; const secureStreak = `${s.secure_streak ?? 0}/${s.secure_streak_required ?? 8}`; $('secureGate').textContent = s.dynamic_adjustment_enabled ? `已放行 · ${secureStreak}` : `${s.secure_evidence ? '稳定确认中' : '等待三指稳定'} · ${secureStreak}`; $('commands').textContent = s.command_count ?? 0; $('sensorAge').textContent = `${Math.round((s.sensor_age || 0) * 1000)} ms`; $('safetyLimits').textContent = (s.safety_pressures || []).slice(0, 3).map((v) => Math.round(v)).join(' / '); $('closedLoopEffort').textContent = (s.closed_loop_efforts || []).slice(0, 3).map((v) => `${Number(v).toFixed(2)}×`).join(' / '); $('thumbAdjustment').textContent = (s.tip_dynamic_adjustments || [s.thumb_dynamic_adjustment ?? 0]).slice(0, 3).map((v) => `${v} raw`).join(' / '); $('footer').textContent = s.error || (s.source === 'hardware' ? '左手触觉数据在线' : '离线自适应模拟在线'); drawMatrices(s); if (s.pose && !previewTimer) applyPose(s.pose); }
async function poll() { try { updateState(await api('/api/state')); } catch (error) { $('footer').textContent = `服务不可用：${error.message}`; } }
buildMatrices(); poll(); setInterval(poll, 500);

// Hardware actions are always explicit and guarded by the server.
$('connectButton').addEventListener('click', async () => { try { await api('/api/hardware/connect', {}); toast('左手已连接'); poll(); } catch (error) { toast(error.message, true); } });
$('armButton').addEventListener('click', async () => { if (!window.confirm('确认现场净空、急停可触达，并解锁番茄实机控制？')) return; try { await api('/api/actual/arm', {confirm:'TOMATO_LEFT_HAND_CONTROL'}); toast('实机控制已解锁'); poll(); } catch (error) { toast(error.message, true); } });
$('disarmButton').addEventListener('click', async () => { try { await api('/api/actual/disarm', {}); toast('实机控制已失能'); poll(); } catch (error) { toast(error.message, true); } });
$('releaseButton').addEventListener('click', async () => { try { await api('/api/release', {}); toast('已请求复位张开'); } catch (error) { toast(error.message, true); } });
$('stopButton').addEventListener('click', async () => { try { await api('/api/tomato/emergency-stop', {}); token = ''; $('startButton').disabled = true; toast('急停已执行并复位张开'); } catch (error) { toast(error.message, true); } });
