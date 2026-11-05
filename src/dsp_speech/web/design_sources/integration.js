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
    cycleTimer = window.setInterval(cycleRows, 4800);
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

  // The speaker page uses the same live microphone path, but posts to its own MFCC+DTW endpoint.
  const speakerScope = document.getElementById("page-speaker") || document.getElementById("screen-speaker");
  if (speakerScope) {
    let speakerPanel = speakerScope.querySelector(".panel");
    let speakerWave = speakerScope.querySelector(".waveform-container");
    let speakerControls = speakerScope.querySelector(".controls");
    let speakerResult = speakerScope.querySelector(".speaker-id");
    let speakerMeta = speakerScope.querySelector(".card-meta");
    let speakerList = speakerScope.querySelector(".comparison-list");
    if (!speakerPanel) {
      const content = speakerScope.querySelector(".page-content");
      if (content) {
        content.innerHTML = `<h1 class="large-title">说话人识别</h1><p>对着麦克风说一段话，停止后查看匹配结果。</p><section class="panel-box"><div id="speaker-wave" class="wave-container"></div><div class="mobile-mic-controls"><label>输入设备</label><select id="speaker-mic"><option value="">系统默认麦克风</option></select><button id="speaker-refresh" type="button">授权 / 刷新</button></div><p id="speaker-hint" aria-live="polite">首次使用请授权麦克风。</p><button id="speaker-start" type="button">开始录音</button><button id="speaker-stop" type="button" disabled>停止并识别</button><p id="speaker-status">等待录音</p></section><section class="result-card"><h2 id="speaker-result">等待录音</h2><p id="speaker-meta"></p><div id="speaker-scores"></div></section>`;
      }
    }
    const wave = speakerWave || speakerScope.querySelector("#speaker-wave");
    const start = speakerScope.querySelector("#speaker-start") || speakerControls?.querySelector(".btn:first-child");
    const stop = speakerScope.querySelector("#speaker-stop") || speakerControls?.querySelector(".btn:nth-child(2)");
    let select = speakerScope.querySelector("#speaker-mic");
    let hint = speakerScope.querySelector("#speaker-hint") || speakerScope.querySelector(".mic-hint");
    let status = speakerScope.querySelector("#speaker-status") || speakerScope.querySelector(".status-tag");
    const result = speakerResult || speakerScope.querySelector("#speaker-result");
    const meta = speakerMeta || speakerScope.querySelector("#speaker-meta");
    const scoresBox = speakerList || speakerScope.querySelector("#speaker-scores");
    if (wave && start && stop && result && scoresBox) {
      start.id = "speaker-start"; stop.id = "speaker-stop"; stop.disabled = true;
      if (status) status.textContent = "● 等待录音";
      if (meta) meta.textContent = "四位组员数据已载入；开始录音后显示匹配结果。";
      scoresBox.replaceChildren();
      if (!select) {
        const controls = document.createElement("div"); controls.className="speaker-device-controls";controls.style.cssText="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:12px";
        controls.innerHTML='<label for="speaker-mic">输入设备</label><select id="speaker-mic"><option value="">系统默认麦克风</option></select><button id="speaker-refresh" type="button">授权 / 刷新</button><span id="speaker-hint" aria-live="polite">允许麦克风权限后可选择耳机麦克风。</span>';
        (speakerControls || speakerPanel).append(controls); select=controls.querySelector("select");hint=controls.querySelector("span");
      }
      if (result.classList.contains("speaker-id")) result.textContent = "—";
      if (select) select.id = "speaker-mic";
      const canvas = document.createElement("canvas"); canvas.style.cssText = "width:100%;height:100%;display:block"; wave.replaceChildren(canvas);
      const ctx = canvas.getContext("2d"); let stream=null, ac=null, proc=null, source=null, chunks=[];
      const draw=(samples)=>{const r=canvas.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);if(!r.width||!r.height)return;canvas.width=r.width*d;canvas.height=r.height*d;ctx.setTransform(d,0,0,d,0,0);ctx.clearRect(0,0,r.width,r.height);ctx.strokeStyle="#173b62";ctx.beginPath();ctx.moveTo(0,r.height/2);ctx.lineTo(r.width,r.height/2);if(samples.length){for(let x=0;x<r.width;x++){const i=Math.floor(x*samples.length/r.width),v=Math.max(-1,Math.min(1,samples[i]||0))*r.height*.42;ctx.moveTo(x,r.height/2-v);ctx.lineTo(x,r.height/2+v)}}ctx.stroke()};
      const refresh=async()=>{try{const dev=(await navigator.mediaDevices.enumerateDevices()).filter(x=>x.kind==="audioinput"),old=select?.value;if(select){select.replaceChildren(new Option("系统默认麦克风",""));dev.forEach((x,i)=>select.add(new Option(x.label||`麦克风 ${i+1}`,x.deviceId)));if([...select.options].some(x=>x.value===old))select.value=old}}catch(e){if(hint)hint.textContent=e.message}};
      const authorize=async()=>{try{const s=await navigator.mediaDevices.getUserMedia({audio:true});s.getTracks().forEach(t=>t.stop());await refresh()}catch(e){if(hint)hint.textContent=`麦克风权限失败：${e.message}`}};
      const refreshBtn=speakerScope.querySelector("#speaker-refresh");refreshBtn?.addEventListener("click",authorize);
      start.addEventListener("click",async()=>{try{chunks=[];const constraints={channelCount:1,echoCancellation:true,noiseSuppression:true};if(select?.value)constraints.deviceId={exact:select.value};stream=await navigator.mediaDevices.getUserMedia({audio:constraints});ac=new(window.AudioContext||window.webkitAudioContext)();await ac.resume();source=ac.createMediaStreamSource(stream);proc=ac.createScriptProcessor(4096,1,1);proc.onaudioprocess=e=>{chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));e.outputBuffer.getChannelData(0).fill(0);draw(chunks[chunks.length-1])};source.connect(proc);proc.connect(ac.destination);start.disabled=true;stop.disabled=false;if(status)status.textContent="正在录音（最长8秒）";if(hint)hint.textContent="说一段自然语音，再点击停止并识别。";setTimeout(()=>{if(ac)stop.click()},8000)}catch(e){if(hint)hint.textContent=`无法开始录音：${e.message}`}});
      stop.addEventListener("click",async()=>{if(!ac)return;const rate=ac.sampleRate;proc.disconnect();source.disconnect();stream.getTracks().forEach(t=>t.stop());await ac.close();ac=null;const n=chunks.reduce((s,a)=>s+a.length,0),samples=new Float32Array(n);let off=0;chunks.forEach(a=>{samples.set(a,off);off+=a.length});draw(samples);start.disabled=false;stop.disabled=true;if(status)status.textContent="正在识别…";try{const res=await fetch("/speaker/recognize",{method:"POST",headers:{"Content-Type":"application/octet-stream","X-Sample-Rate":String(rate)},body:samples.buffer}),data=await res.json();if(!res.ok)throw Error(data.error||"识别失败");result.textContent=data.speaker_name; if(meta)meta.textContent=`最小 DTW 距离 ${Number(data.distance).toFixed(2)} · 有效语音 ${Number(data.audio.speech_seconds).toFixed(2)} 秒`;const entries=Object.entries(data.scores);scoresBox.replaceChildren(...entries.map(([sid,v])=>{const row=document.createElement("div");row.className="comparison-item";row.innerHTML=`<div class="comp-header"><span></span><span></span></div><div class="comp-bar-bg"><div class="comp-bar-fill"></div></div>`;row.querySelector(".comp-header span:first-child").textContent=v.name;row.querySelector(".comp-header span:last-child").textContent=Number(v.distance).toFixed(2);row.querySelector(".comp-bar-fill").style.width=`${Math.max(4,100*v.distance/Math.max(...entries.map(([,x])=>x.distance)))}%`;return row}));if(status)status.textContent="识别完成"}catch(e){if(status)status.textContent="识别失败";if(hint)hint.textContent=e.message}});
      refresh();
    }
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
