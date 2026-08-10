const $ = (id) => document.getElementById(id);

const elements = {
  deviceStatus: $("device-status"),
  deviceStatusLabel: $("device-status-label"),
  streamState: $("stream-state"),
  streamStateLabel: $("stream-state-label"),
  usbLabel: $("usb-label"),
  settingsButton: $("settings-button"),
  settingsDialog: $("settings-dialog"),
  settingsError: $("settings-error"),
  saveSettings: $("save-settings"),
  retryButton: $("retry-button"),
  depthRange: $("depth-range"),
  depthRangeValue: $("depth-range-value"),
  quality: $("quality"),
  qualityValue: $("quality-value"),
  depthMaxLabel: $("depth-max-label"),
};

const streamElements = {
  rgb: {
    frame: $("rgb-frame"),
    resolution: $("rgb-resolution"),
    fps: $("rgb-fps"),
    time: $("rgb-time"),
    overlayTitle: $("rgb-overlay-title"),
    overlayDetail: $("rgb-overlay-detail"),
  },
  depth: {
    frame: $("depth-frame"),
    resolution: $("depth-resolution"),
    fps: $("depth-fps"),
    time: $("depth-time"),
    overlayTitle: $("depth-overlay-title"),
    overlayDetail: $("depth-overlay-detail"),
  },
};

let latestSettings = {
  depth_max_mm: 5000,
  jpeg_quality: 84,
};

function shortTime(value) {
  if (!value) return "尚无画面";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

function streamMessage(name, stream) {
  if (stream.state === "error") {
    return {
      title: name === "rgb" ? "RGB 暂不可用" : "深度暂不可用",
      detail: stream.error || "正在自动重试",
    };
  }
  return {
    title: name === "rgb" ? "正在连接 RGB" : "正在连接深度传感器",
    detail: stream.error || (name === "rgb" ? "等待 UVC 视频节点" : "等待 Orbbec SDK"),
  };
}

function renderStream(name, stream) {
  const view = streamElements[name];
  view.frame.classList.toggle("live", stream.live);
  view.frame.classList.toggle("error", stream.state === "error");
  view.resolution.textContent =
    stream.width && stream.height ? `${stream.width} × ${stream.height}` : "— × —";
  view.fps.textContent = stream.fps ? `${stream.fps.toFixed(1)} FPS` : "— FPS";
  view.time.textContent = shortTime(stream.timestamp);
  view.time.dateTime = stream.timestamp || "";

  const message = streamMessage(name, stream);
  view.overlayTitle.textContent = message.title;
  view.overlayDetail.textContent = message.detail;
}

function renderStatus(status) {
  const { state, device, streams, settings } = status;
  elements.deviceStatus.className = `device-status ${state}`;
  elements.streamState.className = `status-bar-item stream-state ${
    state === "connected" ? "live" : state
  }`;

  if (state === "connected") {
    elements.deviceStatusLabel.textContent = "设备已连接";
    elements.streamStateLabel.textContent = "正在推流";
  } else if (state === "partial") {
    elements.deviceStatusLabel.textContent = "部分画面可用";
    elements.streamStateLabel.textContent = "单路推流";
  } else {
    elements.deviceStatusLabel.textContent = "设备未连接";
    elements.streamStateLabel.textContent = "等待视频流";
  }

  elements.usbLabel.textContent = device.connection || "USB";

  renderStream("rgb", streams.rgb);
  renderStream("depth", streams.depth);

  latestSettings = settings;
  elements.depthMaxLabel.textContent = `${settings.depth_max_mm} mm`;
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderStatus(await response.json());
  } catch (error) {
    elements.deviceStatus.className = "device-status disconnected";
    elements.deviceStatusLabel.textContent = "服务连接中断";
    elements.streamState.className = "status-bar-item stream-state";
    elements.streamStateLabel.textContent = "正在重连网页服务";
  }
}

function syncSettingOutputs() {
  elements.depthRangeValue.textContent = `${elements.depthRange.value} mm`;
  elements.qualityValue.textContent = `${elements.quality.value}%`;
}

elements.settingsButton.addEventListener("click", () => {
  elements.depthRange.value = latestSettings.depth_max_mm;
  elements.quality.value = latestSettings.jpeg_quality;
  elements.settingsError.textContent = "";
  syncSettingOutputs();
  elements.settingsDialog.showModal();
});

elements.depthRange.addEventListener("input", syncSettingOutputs);
elements.quality.addEventListener("input", syncSettingOutputs);

elements.saveSettings.addEventListener("click", async () => {
  elements.saveSettings.disabled = true;
  elements.settingsError.textContent = "";
  try {
    const response = await fetch("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        depth_max_mm: Number(elements.depthRange.value),
        jpeg_quality: Number(elements.quality.value),
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "保存失败");
    latestSettings = payload.settings;
    elements.depthMaxLabel.textContent = `${latestSettings.depth_max_mm} mm`;
    elements.settingsDialog.close();
  } catch (error) {
    elements.settingsError.textContent = error.message;
  } finally {
    elements.saveSettings.disabled = false;
  }
});

elements.retryButton.addEventListener("click", async () => {
  elements.retryButton.disabled = true;
  elements.settingsError.textContent = "";
  try {
    const response = await fetch("/api/retry", { method: "POST" });
    if (!response.ok) throw new Error("重连请求失败");
    renderStatus((await response.json()).status);
    elements.settingsDialog.close();
  } catch (error) {
    elements.settingsError.textContent = error.message;
  } finally {
    elements.retryButton.disabled = false;
  }
});

refreshStatus();
setInterval(refreshStatus, 1000);
