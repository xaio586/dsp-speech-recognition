(() => {
  "use strict";

  const routeByScreen = {
    "screen-overview": "/",
    "screen-speaker": "/speaker.html",
    "screen-digit": "/digits.html",
    "screen-docs": "/docs.html",
  };
  const screenByPath = Object.fromEntries(Object.entries(routeByScreen).map(([screen, path]) => [path, screen]));
  screenByPath["/index.html"] = "screen-overview";
  screenByPath["/templates.html"] = "screen-docs";
  const screens = [...document.querySelectorAll("#app-container .screen")];
  if (screens.length) window.stopCycle?.();

  const fadeMs = 240;
  let activeScreenId = screenByPath[window.location.pathname] || "screen-overview";
  let isTransitioning = false;
  if (screens.length) {

  function setInitialScreen() {
    screens.forEach((screen) => {
      const isActive = screen.id === activeScreenId;
      screen.classList.toggle("active", isActive);
      screen.classList.remove("is-entering", "is-leaving");
      screen.classList.toggle("hidden-right", !isActive);
      if (isActive) screen.classList.remove("hidden-left");
      screen.setAttribute("aria-hidden", String(!isActive));
    });
    document.documentElement.removeAttribute("data-start-view");
    window.currentScreenId = activeScreenId;
  }

  function pathFor(screenId) {
    return routeByScreen[screenId] || "/";
  }

  function navigateTo(screenId, options = {}) {
    const target = document.getElementById(screenId);
    const current = document.getElementById(activeScreenId);
    if (!target || screenId === activeScreenId || isTransitioning) return;

    const { updateHistory = true } = options;
    isTransitioning = true;
    if (updateHistory) {
      const targetPath = pathFor(screenId);
      if (window.location.pathname !== targetPath) {
        window.history.pushState({ screenId }, "", targetPath);
      }
    }

    current?.classList.remove("active");
    if (current) {
      current.classList.remove("is-entering");
      current.classList.add("is-leaving");
      current.setAttribute("aria-hidden", "true");
    }

    target.classList.remove("hidden-left", "hidden-right", "is-leaving");
    target.classList.add("is-entering");
    target.setAttribute("aria-hidden", "false");
    void target.offsetWidth;
    requestAnimationFrame(() => {
      target.classList.remove("is-entering");
      target.classList.add("active");
    });

    const previousId = activeScreenId;
    activeScreenId = screenId;
    window.currentScreenId = screenId;
    window.setTimeout(() => {
      const previous = document.getElementById(previousId);
      previous?.classList.remove("is-leaving");
      previous?.classList.add("hidden-right");
      isTransitioning = false;
    }, fadeMs + 30);

    if (screenId === "screen-overview") startCycle();
    else stopCycle();

    if (window.digitRecorder?.isRecording()) window.digitRecorder.stopWithoutRecognition();
  }

  window.navigateTo = navigateTo;
  window.showPage = (pageId) => {
    const screenId = {
      "page-home": "screen-overview",
      "page-speaker": "screen-speaker",
      "page-digits": "screen-digit",
      "page-docs": "screen-docs",
    }[pageId];
    if (screenId) navigateTo(screenId);
  };
  window.addEventListener("popstate", () => {
    const screenId = screenByPath[window.location.pathname] || "screen-overview";
    navigateTo(screenId, { updateHistory: false });
  });
  setInitialScreen();

  // The source design has a decorative rotating row showcase; keep it still while a row is touched.
  const overviewRows = [...document.querySelectorAll("#screen-overview .list-row")];
  let expandedIdx = 0;
  let cycleTimer = null;
  function cycleRows() {
    if (!overviewRows.length) return;
    expandedIdx = (expandedIdx + 1) % overviewRows.length;
    overviewRows.forEach((row, index) => row.classList.toggle("expanded", index === expandedIdx));
  }
  function startCycle() {
    stopCycle();
    if (!overviewRows.length || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    cycleTimer = window.setInterval(cycleRows, 3200);
  }
  function stopCycle() {
    if (cycleTimer) window.clearInterval(cycleTimer);
    cycleTimer = null;
  }
  overviewRows.forEach((row, index) => {
    const selectRow = () => {
      stopCycle();
      expandedIdx = index;
      overviewRows.forEach((item, currentIndex) => item.classList.toggle("expanded", currentIndex === index));
    };
    row.addEventListener("pointerdown", selectRow, { passive: true });
  });
  if (activeScreenId === "screen-overview") startCycle();
  } else {
    const desktopPageByPath = {
      "/": "page-home", "/index.html": "page-home", "/speaker.html": "page-speaker",
      "/digits.html": "page-digits", "/docs.html": "page-docs", "/templates.html": "page-docs",
    };
    const initialPage = document.documentElement.dataset.startPage || desktopPageByPath[window.location.pathname] || "page-home";
    if (initialPage !== "page-home") window.showPage?.(initialPage);
  }

  const startButton = document.getElementById("digit-start");
  const stopButton = document.getElementById("digit-stop");
  const micSelect = document.getElementById("mic-select");
  const micRefresh = document.getElementById("mic-refresh");
  const micHint = document.getElementById("mic-hint");
  const micStatus = document.getElementById("mic-status");
  const waveformContainer = document.getElementById("wave-digit");
  const inputFormat = document.getElementById("input-format");
  const resultDigit = document.getElementById("result-digit");
  const resultMeta = document.getElementById("result-meta");
  const scoreList = document.getElementById("score-list");
  if (!startButton || !stopButton || !waveformContainer) return;

  const scoreRows = new Map();
  const desktopScores = scoreList.classList.contains("dtw-list");
  for (let digit = 0; digit <= 9; digit += 1) {
    const row = document.createElement("div");
    row.className = desktopScores ? "dtw-row score-row" : "chart-item score-row";
    row.hidden = true;
    row.innerHTML = desktopScores
      ? `<span class="dtw-label score-label">${digit}</span><div class="dtw-bar-bg"><div class="dtw-bar-fill score-fill"></div></div><span class="dtw-value score-value">—</span>`
      : `<div class="chart-labels"><span class="score-label">数字 ${digit}</span><span class="score-value">—</span></div><div class="chart-bar-bg"><div class="chart-bar-fill score-fill"></div></div>`;
    scoreList.appendChild(row);
    scoreRows.set(String(digit), row);
  }

  const canvas = document.createElement("canvas");
  canvas.setAttribute("aria-label", "麦克风录音波形");
  canvas.style.cssText = "display:block;width:100%;height:100%;";
  waveformContainer.replaceChildren(canvas);
  let audioContext = null;
  let stream = null;
  let sourceNode = null;
  let processorNode = null;
  let chunks = [];
  let recordingStartedAt = 0;
  let timerId = null;
  let recordedSamples = new Float32Array(0);

  function setStatus(text) {
    micStatus.textContent = text;
  }

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(rect.width * ratio));
    canvas.height = Math.max(1, Math.round(rect.height * ratio));
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    drawWave(recordedSamples);
  }

  function drawWave(samples) {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(rect.width * ratio) || canvas.height !== Math.round(rect.height * ratio)) {
      canvas.width = Math.max(1, Math.round(rect.width * ratio));
      canvas.height = Math.max(1, Math.round(rect.height * ratio));
    }
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, rect.width, rect.height);
    const mid = rect.height / 2;
    context.strokeStyle = "rgba(23,59,98,.38)";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(0, mid + .5);
    context.lineTo(rect.width, mid + .5);
    context.stroke();
    if (!samples.length) return;
    let peak = 0;
    for (let i = 0; i < samples.length; i += 1) peak = Math.max(peak, Math.abs(samples[i]));
    const scale = Math.min(rect.height * .38 / Math.max(peak, 1e-6), 900);
    context.strokeStyle = "#173b62";
    context.lineWidth = 1;
    context.beginPath();
    for (let x = 0; x < Math.ceil(rect.width); x += 1) {
      const from = Math.floor(x / rect.width * samples.length);
      const to = Math.max(from + 1, Math.floor((x + 1) / rect.width * samples.length));
      let min = 0;
      let max = 0;
      for (let i = from; i < Math.min(to, samples.length); i += 1) {
        min = Math.min(min, samples[i]);
        max = Math.max(max, samples[i]);
      }
      context.moveTo(x + .5, mid - max * scale);
      context.lineTo(x + .5, mid - min * scale);
    }
    context.stroke();
  }

  function combineChunks() {
    const length = chunks.reduce((total, item) => total + item.length, 0);
    const output = new Float32Array(length);
    let offset = 0;
    chunks.forEach((item) => { output.set(item, offset); offset += item.length; });
    return output;
  }

  async function refreshMicrophones() {
    if (!navigator.mediaDevices?.enumerateDevices) {
      micHint.textContent = "当前浏览器不支持麦克风设备列表。";
      return;
    }
    try {
      const previous = micSelect.value;
      const inputs = (await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === "audioinput");
      micSelect.replaceChildren(new Option("系统默认麦克风", ""));
      inputs.forEach((device, index) => {
        micSelect.add(new Option(device.label || `麦克风 ${index + 1}（授权后显示名称）`, device.deviceId));
      });
      if ([...micSelect.options].some((option) => option.value === previous)) micSelect.value = previous;
      if (inputs.some((device) => device.label)) micHint.textContent = "已读取设备；可选择手机或耳机麦克风。";
      else micHint.textContent = "设备名称需先授权后显示。";
    } catch (error) {
      micHint.textContent = `读取麦克风列表失败：${error.message}`;
    }
  }

  async function authorizeMicrophones() {
    try {
      const permissionStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      permissionStream.getTracks().forEach((track) => track.stop());
      await refreshMicrophones();
    } catch (error) {
      micHint.textContent = error.name === "NotAllowedError"
        ? "请在浏览器提示中允许麦克风权限。"
        : `无法访问麦克风：${error.message}`;
    }
  }

  async function releaseMicrophone() {
    if (timerId) window.clearInterval(timerId);
    timerId = null;
    if (processorNode) {
      processorNode.onaudioprocess = null;
      processorNode.disconnect();
    }
    processorNode = null;
    if (sourceNode) sourceNode.disconnect();
    sourceNode = null;
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = null;
    if (audioContext) await audioContext.close();
    audioContext = null;
  }

  function clearResult() {
    resultDigit.textContent = "—";
    resultMeta.textContent = "等待本次录音结果";
    document.querySelectorAll(".digit[data-digit]").forEach((item) => item.classList.remove("active"));
    scoreRows.forEach((row) => {
      row.classList.remove("is-best");
      row.querySelector(".score-value").textContent = "—";
      row.querySelector(".score-fill").style.width = "0";
    });
  }

  function showScores(scores) {
    const entries = Object.entries(scores).sort((a, b) => Number(a[1]) - Number(b[1]));
    const min = Number(entries[0]?.[1] ?? 0);
    const max = Number(entries.at(-1)?.[1] ?? min);
    const spread = Math.max(max - min, 1e-9);
    entries.forEach(([digit, value], index) => {
      const row = scoreRows.get(digit);
      if (!row) return;
      row.hidden = index >= 3;
      row.querySelector(".score-value").textContent = Number(value).toFixed(2);
      const distanceScale = 12 + ((Number(value) - min) / spread) * 88;
      row.querySelector(".score-fill").style.width = `${distanceScale}%`;
      row.classList.toggle("is-best", index === 0);
    });
  }

  async function startRecording() {
    if (!navigator.mediaDevices?.getUserMedia) {
      micHint.textContent = "请使用支持麦克风的手机浏览器，并通过 HTTPS 打开网页。";
      return;
    }
    clearResult();
    micHint.textContent = "";
    try {
      const audio = { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true };
      if (micSelect.value) audio.deviceId = { exact: micSelect.value };
      stream = await navigator.mediaDevices.getUserMedia({ audio });
      await refreshMicrophones();
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      await audioContext.resume();
      if (inputFormat) inputFormat.textContent = `${audioContext.sampleRate} Hz / mono`;
      sourceNode = audioContext.createMediaStreamSource(stream);
      processorNode = audioContext.createScriptProcessor(4096, 1, 1);
      chunks = [];
      recordedSamples = new Float32Array(0);
      drawWave(recordedSamples);
      processorNode.onaudioprocess = (event) => {
        chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
        event.outputBuffer.getChannelData(0).fill(0);
        if (chunks.length % 2 === 0) {
          recordedSamples = combineChunks();
          drawWave(recordedSamples);
        }
      };
      sourceNode.connect(processorNode);
      processorNode.connect(audioContext.destination);
      recordingStartedAt = Date.now();
      startButton.disabled = true;
      stopButton.disabled = false;
      setStatus("● 正在录音 00:00");
      timerId = window.setInterval(() => {
        const seconds = Math.floor((Date.now() - recordingStartedAt) / 1000);
        setStatus(`● 正在录音 00:${String(seconds).padStart(2, "0")}`);
        if (seconds >= 8) stopRecording();
      }, 200);
    } catch (error) {
      await releaseMicrophone();
      startButton.disabled = false;
      stopButton.disabled = true;
      setStatus("● 等待录音");
      micHint.textContent = error.name === "NotAllowedError"
        ? "麦克风权限未开放，请允许浏览器使用麦克风。"
        : `录音启动失败：${error.message}`;
    }
  }

  async function stopRecording({ recognize = true } = {}) {
    if (!audioContext) return;
    const sampleRate = audioContext.sampleRate;
    await releaseMicrophone();
    recordedSamples = combineChunks();
    drawWave(recordedSamples);
    startButton.disabled = false;
    stopButton.disabled = true;
    if (!recognize) {
      setStatus("● 录音已停止");
      return;
    }
    if (recordedSamples.length < sampleRate * .2) {
      setStatus("● 录音太短");
      micHint.textContent = "请完整说完一个数字再停止录音。";
      return;
    }
    setStatus("● 正在识别");
    try {
      const response = await fetch("/recognize", {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream", "X-Sample-Rate": String(sampleRate) },
        body: recordedSamples.buffer,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "识别失败，请重新录音。 ");
      resultDigit.textContent = payload.digit;
      resultMeta.textContent = `最小 DTW 距离 ${Number(payload.distance).toFixed(3)} · 有效语音 ${Number(payload.audio.speech_seconds).toFixed(2)} 秒`;
      document.querySelectorAll(".digit[data-digit]").forEach((item) => item.classList.toggle("active", item.dataset.digit === payload.digit));
      showScores(payload.scores);
      setStatus("● 识别完成");
      micHint.textContent = "如果数字不对，可换一个麦克风位置或重新说一遍。";
    } catch (error) {
      setStatus("● 识别失败");
      micHint.textContent = error.message;
    }
  }

  startButton.addEventListener("click", startRecording);
  stopButton.addEventListener("click", () => stopRecording());
  micRefresh.addEventListener("click", authorizeMicrophones);
  micSelect.addEventListener("change", () => {
    const option = micSelect.selectedOptions[0];
    micHint.textContent = option?.value ? `已选择：${option.textContent}` : "已选择系统默认麦克风。";
  });
  window.addEventListener("resize", resizeCanvas, { passive: true });
  if (window.ResizeObserver) new ResizeObserver(resizeCanvas).observe(waveformContainer);
  navigator.mediaDevices?.addEventListener?.("devicechange", refreshMicrophones);
  resizeCanvas();
  refreshMicrophones();

  window.digitRecorder = {
    isRecording: () => Boolean(audioContext),
    stopWithoutRecognition: () => stopRecording({ recognize: false }),
  };
})();
