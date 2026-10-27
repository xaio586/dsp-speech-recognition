(() => {
  "use strict";

  const routes = {
    "page-home": "/index.html",
    "page-speaker": "/speaker.html",
    "page-digits": "/digits.html",
    "page-docs": "/docs.html",
  };

  let navigating = false;
  function navigateTo(path) {
    if (!path || navigating) return;
    const current = window.location.pathname === "/" ? "/index.html" : window.location.pathname;
    if (current === path) return;
    navigating = true;
    document.body.classList.remove("e3-visible");
    document.body.classList.add("e3-leaving");
    window.setTimeout(() => window.location.assign(path), 280);
  }

  window.showPage = (pageId) => {
    if (routes[pageId]) navigateTo(routes[pageId]);
  };

  document.addEventListener("click", (event) => {
    const anchor = event.target.closest("a[href]");
    if (!anchor) return;
    const href = anchor.getAttribute("href");
    const target = href === "/" ? "/index.html" : href === "index.html" ? "/index.html" : null;
    if (!target) return;
    event.preventDefault();
    navigateTo(target);
  }, true);

  document.querySelectorAll(".back-btn, .back-link").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      navigateTo("/index.html");
    });
  });

  if (window.location.pathname !== "/digits.html") return;

  const buttonGroup = document.querySelector(".button-group");
  const controlSection = document.querySelector(".control-section");
  const buttons = [...document.querySelectorAll(".button-group .btn")];
  const startButton = buttons[0];
  const stopButton = buttons[1];
  const statusPill = document.querySelector(".status-pill");
  const resultValue = document.querySelector(".result-val");
  const resultMeta = document.querySelector(".result-meta");
  const waveformContainer = document.querySelector(".waveform-viz");
  const statRows = [...document.querySelectorAll(".stat-row")];
  const audioLabel = document.querySelector(".viz-section .section-tag");
  if (!buttonGroup || !controlSection || !startButton || !stopButton) return;

  const style = document.createElement("style");
  style.textContent = `
    .viz-section{width:100%;min-width:0}
    .waveform-viz{display:block!important;position:relative!important;width:100%!important;min-width:0!important;max-width:100%!important;height:120px!important;box-sizing:border-box!important;padding:0!important;overflow:hidden!important;flex:none!important}
    .waveform-viz canvas{display:block!important;position:absolute!important;inset:0!important;width:100%!important;height:100%!important;min-width:0!important;max-width:100%!important;box-sizing:border-box!important}
    .e3-mic-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:14px;font:12px Inter,"Noto Sans SC",sans-serif;color:#17201a}
    .e3-mic-row select{max-width:min(360px,55vw);min-width:190px;padding:7px 24px 7px 5px;border:0;border-bottom:1px solid rgba(20,30,25,.5);background:rgba(247,245,235,.28);color:inherit;font:inherit;outline:none}
    .e3-mic-row button{padding:5px 2px;border:0;background:transparent;color:#17574c;font:inherit;cursor:pointer}
    .e3-mic-row button:hover{text-decoration:underline}
    .e3-mic-row .e3-mic-hint{color:rgba(20,30,25,.65);font-size:11px}
    @media(max-width:650px){.e3-mic-row select{min-width:150px;max-width:68vw}}
  `;
  document.head.appendChild(style);

  const micRow = document.createElement("div");
  micRow.className = "e3-mic-row";
  micRow.innerHTML = '<span>输入设备</span><select aria-label="选择麦克风"><option value="">系统默认麦克风</option></select><button type="button">授权 / 刷新设备</button><span class="e3-mic-hint"></span>';
  controlSection.appendChild(micRow);
  const micSelect = micRow.querySelector("select");
  const refreshButton = micRow.querySelector("button");
  const micHint = micRow.querySelector(".e3-mic-hint");

  const waveformCanvas = document.createElement("canvas");
  waveformCanvas.setAttribute("aria-label", "麦克风实时录音波形");
  waveformContainer.style.cssText += ";display:block;position:relative;width:100%;min-width:0;max-width:100%;height:120px;box-sizing:border-box;padding:0;overflow:hidden;flex:none";
  waveformCanvas.style.cssText = "display:block;position:absolute;inset:0;box-sizing:border-box;width:100%;height:100%;min-width:0;max-width:100%";
  waveformContainer.replaceChildren(waveformCanvas);
  const waveformContext = waveformCanvas.getContext("2d");

  let audioContext = null;
  let mediaStream = null;
  let sourceNode = null;
  let processorNode = null;
  let chunks = [];
  let timerId = null;
  let startedAt = 0;
  let recorded = new Float32Array(0);

  const setStatus = (value) => {
    if (statusPill) statusPill.textContent = value;
  };

  async function refreshMicrophones() {
    if (!navigator.mediaDevices?.enumerateDevices) {
      micHint.textContent = "浏览器暂不支持麦克风设备列表";
      return;
    }
    try {
      const devices = (await navigator.mediaDevices.enumerateDevices()).filter((item) => item.kind === "audioinput");
      const previous = micSelect.value;
      micSelect.innerHTML = '<option value="">系统默认麦克风</option>';
      devices.forEach((device, index) => {
        const option = document.createElement("option");
        option.value = device.deviceId;
        option.textContent = device.label || `麦克风 ${index + 1}（授权后显示名称）`;
        micSelect.appendChild(option);
      });
      if ([...micSelect.options].some((option) => option.value === previous)) micSelect.value = previous;
      micHint.textContent = devices.some((device) => device.label)
        ? "选择耳机麦克风后开始录音"
        : "授权后可显示耳机麦克风名称";
    } catch (error) {
      micHint.textContent = `读取设备失败：${error.message}`;
    }
  }

  async function authorizeMicrophones() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((track) => track.stop());
      await refreshMicrophones();
    } catch (error) {
      micHint.textContent = error.name === "NotAllowedError"
        ? "请在浏览器提示中允许麦克风权限"
        : `无法打开麦克风：${error.message}`;
    }
  }

  function joinChunks() {
    const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
    const joined = new Float32Array(length);
    let offset = 0;
    chunks.forEach((chunk) => { joined.set(chunk, offset); offset += chunk.length; });
    return joined;
  }

  function updateWaveform(samples) {
    const bounds = waveformCanvas.getBoundingClientRect();
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, bounds.width);
    const height = Math.max(1, bounds.height);
    waveformCanvas.width = Math.round(width * pixelRatio);
    waveformCanvas.height = Math.round(height * pixelRatio);
    waveformContext.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    waveformContext.clearRect(0, 0, width, height);

    const center = height / 2;
    waveformContext.strokeStyle = "rgba(23,59,98,.46)";
    waveformContext.lineWidth = 1;
    waveformContext.beginPath();
    waveformContext.moveTo(0, center + .5);
    waveformContext.lineTo(width, center + .5);
    waveformContext.stroke();
    if (!samples.length) return;

    let peak = 0;
    for (let i = 0; i < samples.length; i += 1) peak = Math.max(peak, Math.abs(samples[i]));
    const scale = Math.min((height * .38) / Math.max(peak, 1e-6), 800);
    waveformContext.strokeStyle = "#173b62";
    waveformContext.lineWidth = 1;
    waveformContext.beginPath();
    let started = false;
    for (let x = 0; x < Math.ceil(width); x += 1) {
      const from = Math.floor((x / width) * samples.length);
      const to = Math.max(from + 1, Math.floor(((x + 1) / width) * samples.length));
      let minimum = 0;
      let maximum = 0;
      for (let i = from; i < Math.min(to, samples.length); i += 1) {
        minimum = Math.min(minimum, samples[i]);
        maximum = Math.max(maximum, samples[i]);
      }
      const yTop = center - maximum * scale;
      const yBottom = center - minimum * scale;
      if (!started) {
        waveformContext.moveTo(x + .5, yTop);
        started = true;
      }
      waveformContext.lineTo(x + .5, yTop);
      waveformContext.lineTo(x + .5, yBottom);
    }
    waveformContext.stroke();
  }

  function updateScores(scores) {
    const ranked = Object.entries(scores).sort((left, right) => left[1] - right[1]).slice(0, 3);
    const largest = Math.max(...ranked.map((item) => item[1]), 1e-6);
    ranked.forEach(([digit, distance], index) => {
      const row = statRows[index];
      if (!row) return;
      const labels = row.querySelectorAll(".stat-label span");
      if (labels[0]) labels[0].textContent = `数字 ${digit}`;
      if (labels[1]) labels[1].textContent = Number(distance).toFixed(2);
      const fill = row.querySelector(".stat-bar-fill");
      if (fill) fill.style.width = `${Math.max(10, (distance / largest) * 100)}%`;
    });
  }

  async function releaseMicrophone() {
    if (timerId) { clearInterval(timerId); timerId = null; }
    if (processorNode) { processorNode.onaudioprocess = null; processorNode.disconnect(); processorNode = null; }
    if (sourceNode) { sourceNode.disconnect(); sourceNode = null; }
    if (mediaStream) { mediaStream.getTracks().forEach((track) => track.stop()); mediaStream = null; }
    if (audioContext) { const context = audioContext; audioContext = null; await context.close(); }
  }

  async function startRecording() {
    if (!navigator.mediaDevices?.getUserMedia) {
      micHint.textContent = "请通过本机程序页面打开，并使用支持麦克风的浏览器";
      return;
    }
    micHint.textContent = "";
    if (resultValue) resultValue.textContent = "—";
    if (resultMeta) resultMeta.innerHTML = "等待本次录音结果";
    statRows.forEach((row) => {
      const labels = row.querySelectorAll(".stat-label span");
      if (labels[1]) labels[1].textContent = "—";
      const fill = row.querySelector(".stat-bar-fill");
      if (fill) fill.style.width = "0%";
    });
    try {
      const audio = { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true };
      if (micSelect.value) audio.deviceId = { exact: micSelect.value };
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio });
      await refreshMicrophones();
      micHint.textContent = "";
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      sourceNode = audioContext.createMediaStreamSource(mediaStream);
      processorNode = audioContext.createScriptProcessor(4096, 1, 1);
      chunks = [];
      recorded = new Float32Array(0);
      updateWaveform(recorded);
      processorNode.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        chunks.push(new Float32Array(input));
        event.outputBuffer.getChannelData(0).fill(0);
        if (chunks.length % 2 === 0) updateWaveform(joinChunks());
      };
      sourceNode.connect(processorNode);
      processorNode.connect(audioContext.destination);
      if (audioLabel) audioLabel.textContent = `MIC INPUT / WAVEFORM · ${audioContext.sampleRate} Hz / mono`;
      startedAt = Date.now();
      startButton.disabled = true;
      stopButton.disabled = false;
      setStatus("● 正在录音 00:00");
      timerId = window.setInterval(() => {
        const seconds = Math.floor((Date.now() - startedAt) / 1000);
        setStatus(`● 正在录音 00:${String(seconds).padStart(2, "0")}`);
        if (seconds >= 8) stopRecording();
      }, 150);
    } catch (error) {
      await releaseMicrophone();
      startButton.disabled = false;
      stopButton.disabled = true;
      setStatus("● 等待录音");
      micHint.textContent = error.name === "NotAllowedError"
        ? "麦克风权限未开放；请允许浏览器使用麦克风"
        : `录音启动失败：${error.message}`;
    }
  }

  async function stopRecording() {
    if (!audioContext) return;
    const sampleRate = audioContext.sampleRate;
    await releaseMicrophone();
    recorded = joinChunks();
    startButton.disabled = false;
    stopButton.disabled = true;
    updateWaveform(recorded);
    if (recorded.length < sampleRate * 0.2) {
      setStatus("● 录音太短");
      micHint.textContent = "请完整说完一个数字后再停止录音";
      return;
    }
    setStatus("● 正在提取特征并匹配");
    try {
      const response = await fetch("/recognize", {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream", "X-Sample-Rate": String(sampleRate) },
        body: recorded.buffer,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "识别请求失败");
      if (resultValue) resultValue.textContent = payload.digit;
      if (resultMeta) resultMeta.innerHTML = `最小 DTW 距离 ${Number(payload.distance).toFixed(3)}<br>有效语音时长 ${Number(payload.audio.speech_seconds).toFixed(2)} s`;
      updateScores(payload.scores);
      document.querySelectorAll(".digit-node").forEach((node) => {
        node.classList.toggle("active", node.textContent.trim() === payload.digit);
      });
      setStatus("● 识别完成");
    } catch (error) {
      setStatus("● 本次识别失败");
      micHint.textContent = error.message;
    }
  }

  startButton.disabled = false;
  stopButton.disabled = true;
  if (resultValue) resultValue.textContent = "—";
  if (resultMeta) resultMeta.innerHTML = "等待一段新的语音";
  setStatus("● 等待录音");
  startButton.addEventListener("click", startRecording);
  stopButton.addEventListener("click", stopRecording);
  refreshButton.addEventListener("click", authorizeMicrophones);
  window.addEventListener("resize", () => updateWaveform(recorded));
  if (window.ResizeObserver) new ResizeObserver(() => updateWaveform(recorded)).observe(waveformContainer);
  updateWaveform(recorded);
  refreshMicrophones();
  if (navigator.mediaDevices?.addEventListener) {
    navigator.mediaDevices.addEventListener("devicechange", refreshMicrophones);
  }
})();
