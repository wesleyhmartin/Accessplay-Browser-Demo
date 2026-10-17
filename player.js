// --- 1. Declarations ---
const ariaLive = document.getElementById("ariaLive");
const urlParams = new URLSearchParams(window.location.search);
const src = urlParams.get("src") || new URL("demo.mp3", window.location.href).href;

// TARGETING THE DOM VIDEO ELEMENT
const audio = document.getElementById("mainMedia"); 
if (src) {
  audio.src = src;
}

// UI Elements
const playBtn = document.getElementById("playBtn");
const rewindBtn = document.getElementById("rewindBtn");
const forwardBtn = document.getElementById("forwardBtn");
const downloadBtn = document.getElementById("downloadBtn");
const volumeSlider = document.getElementById("volume");
const speedSlider = document.getElementById("speed");
const showCaptions = document.getElementById("showCaptions");
const captionDisplay = document.getElementById("captionDisplay");

const currentTimeSpan = document.getElementById("currentTime");
const durationTimeSpan = document.getElementById("durationTime");
const timeContainer = document.getElementById("timeContainer");
const excludeDomain = document.getElementById("excludeDomain");

let skipSeconds = 10;
let captionWorker;
let captionChunks = [];
let captionLoadStarted = false;
let captionComplete = false;
let captionFirstChunkReceived = false;
let captionGeneration = 0;
let captionRequestController;
let captionAudioContext;
let captionDecodeWatchdog;
let captionResampleWatchdog;
let captionStartFrame;
let captionSkipToChunkIndex = null;
let captionInferenceResolver;
let captionTranscriptionStartOffset = 0;
let captionSettingsLoaded = false;
let captionControlTouched = false;
let playPressedBeforeCaptionSettings = false;
let captionModel = "onnx-community/whisper-tiny.en";
let liveAnnounceTimer;
let liveUpdateTimer;
let liveHideTimer;
chrome.storage.sync.get({
  skipSeconds: 10,
  defaultVolume: 100,
  defaultSpeed: 1,
  captionModel,
  captionsOnByDefault: false
}, (data) => {
  skipSeconds = data.skipSeconds || 10;
  captionModel = data.captionModel && data.captionModel !== "onnx-community/whisper-tiny" ? data.captionModel : captionModel;
  if (!captionControlTouched) showCaptions.checked = data.captionsOnByDefault;
  captionSettingsLoaded = true;

  if (data.captionsOnByDefault && playPressedBeforeCaptionSettings && showCaptions.checked) {
    scheduleCaptionTranscription();
  }
  
  if (data.defaultVolume) {
    const defaultVol = Math.min(100, Math.max(0, data.defaultVolume)) / 100;
    audio.volume = defaultVol;
    volumeSlider.value = data.defaultVolume;
  }
  
  if (data.defaultSpeed) {
    const defaultSpd = Math.min(5, Math.max(0.5, data.defaultSpeed));
    audio.playbackRate = defaultSpd;
    speedSlider.value = defaultSpd.toFixed(1);
  }
});

async function resampleTo16k(buffer, startOffsetSeconds, endOffsetSeconds) {
  const offset = Math.max(0, startOffsetSeconds);
  const end = Math.min(buffer.duration, endOffsetSeconds);
  const targetLength = Math.ceil((end - offset) * 16000);
  if (targetLength <= 0) throw new Error("No audio remains to caption from the current position.");
  const context = new OfflineAudioContext(1, targetLength, 16000);
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);
  source.start(0, offset, end - offset);
  const rendered = await context.startRendering();
  return { samples: rendered.getChannelData(0), offsetSeconds: offset };
}

function downmixAudioChunk(buffer, startSample, endSample) {
  const length = endSample - startSample;
  if (buffer.numberOfChannels === 1) {
    return buffer.getChannelData(0).slice(startSample, endSample);
  }
  const output = new Float32Array(length);
  const scale = 1 / buffer.numberOfChannels;
  for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
    const input = buffer.getChannelData(channel);
    for (let sample = 0; sample < length; sample++) {
      output[sample] += input[startSample + sample] * scale;
    }
  }
  return output;
}

function isCurrentCaptionGeneration(generation) {
  return generation === captionGeneration && showCaptions.checked;
}

function setCaptionLoadingStatus(_message, generation = captionGeneration) {
  if (!isCurrentCaptionGeneration(generation) || captionFirstChunkReceived) return;
  setCaptionDisplayText("Loading...");
}

function finishCaptionLoading(generation) {
  if (generation !== captionGeneration) return;
  if (captionDisplay.textContent === "Loading...") setCaptionDisplayText("");
}

function clearCaptionLoading(generation) {
  if (generation !== captionGeneration) return;
  if (captionDisplay.textContent === "Loading...") setCaptionDisplayText("");
}

function setCaptionDisplayText(text) {
  if (captionDisplay.textContent !== text) captionDisplay.textContent = text;
  captionDisplay.classList.toggle("caption-empty", !text);
}

function hideCaptionLoadingAfterFirstChunk(generation) {
  if (generation !== captionGeneration) return;
  captionFirstChunkReceived = true;
  updateCaptionCue();
}

function cancelCaptionTranscription() {
  cancelAnimationFrame(captionStartFrame);
  captionStartFrame = undefined;
  if (captionLoadStarted && !captionComplete) {
    captionGeneration++;
    captionInferenceResolver?.();
    captionInferenceResolver = undefined;
    captionRequestController?.abort();
    captionWorker?.terminate();
    captionWorker = undefined;
    captionAudioContext?.close().catch(() => {});
    captionAudioContext = undefined;
    clearInterval(captionDecodeWatchdog);
    captionDecodeWatchdog = undefined;
    clearInterval(captionResampleWatchdog);
    captionResampleWatchdog = undefined;
    captionChunks = [];
    captionLoadStarted = false;
    captionFirstChunkReceived = false;
  }
  setCaptionDisplayText("");
}

async function fetchAudioWithProgress(url, signal, generation) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`Audio request failed (${response.status}).`);
  if (!response.body?.getReader) return response.arrayBuffer();
  const reader = response.body.getReader();
  const chunks = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!isCurrentCaptionGeneration(generation)) {
      await reader.cancel();
      return null;
    }
    chunks.push(value);
    setCaptionLoadingStatus("Loading...", generation);
  }
  const combinedAudio = new Blob(chunks, { type: response.headers.get("content-type") || "audio/mpeg" });
  setCaptionLoadingStatus("Loading...", generation);
  return combinedAudio.arrayBuffer();
}

function startCaptionTranscription() {
  if (!src || !showCaptions.checked || captionLoadStarted || captionComplete) return;
  const requestedStartOffset = audio.currentTime;
  captionLoadStarted = true;
  const generation = ++captionGeneration;
  captionRequestController = new AbortController();
  captionFirstChunkReceived = false;
  captionSkipToChunkIndex = null;
  captionTranscriptionStartOffset = requestedStartOffset;
  setCaptionDisplayText("");
  setCaptionLoadingStatus("Loading...", generation);
  try {
    captionWorker = new Worker(chrome.runtime.getURL("captions-worker.bundle.js"));
  } catch (error) {
    captionLoadStarted = false;
    clearCaptionLoading(generation);
    setCaptionDisplayText(`Captions unavailable: ${error.message}`);
    return;
  }
  const failCaptionLoad = (message) => {
    if (!isCurrentCaptionGeneration(generation)) return;
    captionLoadStarted = false;
    captionRequestController?.abort();
    captionWorker?.terminate();
    captionWorker = undefined;
    clearInterval(captionDecodeWatchdog);
    captionDecodeWatchdog = undefined;
    clearInterval(captionResampleWatchdog);
    captionResampleWatchdog = undefined;
    captionAudioContext?.close().catch(() => {});
    captionAudioContext = undefined;
    clearCaptionLoading(generation);
    setCaptionDisplayText(`Captions unavailable: ${message}`);
    captionGeneration++;
  };
  captionWorker.onmessage = ({ data }) => {
    if (!isCurrentCaptionGeneration(generation)) return;
    if (data.type === "status") {
      setCaptionLoadingStatus("Loading...", generation);
    } else if (data.type === "ready") {
      console.info("[AccessPlay Captions] Model is ready; awaiting audio decoding");
      setCaptionLoadingStatus("Loading...", generation);
    } else if (data.type === "model") {
      console.info("[AccessPlay Captions] Model initialization", data);
      setCaptionLoadingStatus("Loading...", generation);
    } else if (data.type === "inference") {
      console.info("[AccessPlay Captions] Model inference", data);
      if (data.status === "completed") {
        captionInferenceResolver?.();
        captionInferenceResolver = undefined;
      }
    } else if (data.type === "stage") {
      setCaptionLoadingStatus("Loading...", generation);
    } else if (data.type === "chunks") {
      captionChunks.push(...data.chunks);
      updateCaptionCue();
      if (data.chunks?.length) hideCaptionLoadingAfterFirstChunk(generation);
      else setCaptionLoadingStatus("Loading...", generation);
    } else if (data.type === "complete") {
      captionLoadStarted = false;
      captionComplete = true;
      finishCaptionLoading(generation);
      if (captionChunks.length) updateCaptionCue();
      else {
        setCaptionDisplayText("No speech was detected.");
      }
    } else if (data.type === "error") {
      failCaptionLoad(data.message);
    }
  };
  captionWorker.onerror = (event) => {
    event.preventDefault();
    console.error("[AccessPlay Captions] Caption worker failed to load or crashed", event.message);
    failCaptionLoad(event.message || "Caption worker failed to start.");
  };
  captionWorker.onmessageerror = () => {
    failCaptionLoad("Caption worker returned an unreadable message.");
  };
  captionWorker.postMessage({ type: "init", model: captionModel });
  setCaptionLoadingStatus("Loading...", generation);
  fetchAudioWithProgress(src, captionRequestController.signal, generation).then((audioBytes) => {
    if (!audioBytes || !isCurrentCaptionGeneration(generation)) return null;
    setCaptionLoadingStatus("Loading...", generation);
    try {
      captionAudioContext = new AudioContext({ sampleRate: 16000 });
    } catch {
      captionAudioContext = new AudioContext();
    }
    const decodeStarted = performance.now();
    console.info("[AccessPlay Captions] Audio decoding started", {
      bytes: audioBytes.byteLength,
      requestedSampleRate: captionAudioContext.sampleRate
    });
    captionDecodeWatchdog = setInterval(() => {
      console.info("[AccessPlay Captions] Audio decoding still pending", { elapsedMs: Math.round(performance.now() - decodeStarted) });
    }, 10000);
    return captionAudioContext.decodeAudioData(audioBytes).then((decoded) => {
      clearInterval(captionDecodeWatchdog);
      captionDecodeWatchdog = undefined;
      console.info("[AccessPlay Captions] Audio decoding completed", {
        elapsedMs: Math.round(performance.now() - decodeStarted),
        durationSeconds: decoded.duration,
        sampleRate: decoded.sampleRate,
        channels: decoded.numberOfChannels
      });
      captionAudioContext.close().catch(() => {});
      captionAudioContext = undefined;
      return decoded;
    }, (error) => {
      clearInterval(captionDecodeWatchdog);
      captionDecodeWatchdog = undefined;
      console.error("[AccessPlay Captions] Audio decoding rejected", {
        elapsedMs: Math.round(performance.now() - decodeStarted),
        error
      });
      throw error;
    });
  }).then(async (decoded) => {
    if (!decoded || !isCurrentCaptionGeneration(generation)) return;
    const chunkSeconds = 30;
    const startOffset = requestedStartOffset >= decoded.duration ? 0 : Math.max(0, requestedStartOffset);
    const totalChunks = Math.ceil((decoded.duration - startOffset) / chunkSeconds);
    if (totalChunks <= 0) throw new Error("No audio remains to caption from the current position.");
    console.info("[AccessPlay Captions] Audio preparation started", {
      durationSeconds: decoded.duration,
      sampleRate: decoded.sampleRate,
      channels: decoded.numberOfChannels,
      chunks: totalChunks
    });
    let chunkIndex = 0;
    while (chunkIndex < totalChunks) {
      if (!isCurrentCaptionGeneration(generation)) return;
      if (captionSkipToChunkIndex !== null) {
        chunkIndex = Math.max(chunkIndex, captionSkipToChunkIndex);
        captionSkipToChunkIndex = null;
        if (chunkIndex >= totalChunks) break;
      }
      const offsetSeconds = startOffset + chunkIndex * chunkSeconds;
      const endOffsetSeconds = Math.min(decoded.duration, offsetSeconds + chunkSeconds);
      let samples;
      if (decoded.sampleRate === 16000) {
        const startSample = Math.round(offsetSeconds * 16000);
        const endSample = Math.min(decoded.length, Math.round(endOffsetSeconds * 16000));
        samples = downmixAudioChunk(decoded, startSample, endSample);
      } else {
        setCaptionLoadingStatus("Loading...", generation);
        const resampleStarted = performance.now();
        captionResampleWatchdog = setInterval(() => {
          console.info("[AccessPlay Captions] Audio chunk resampling still pending", {
            chunk: chunkIndex + 1,
            elapsedMs: Math.round(performance.now() - resampleStarted)
          });
        }, 10000);
        try {
          ({ samples } = await resampleTo16k(decoded, offsetSeconds, endOffsetSeconds));
        } finally {
          clearInterval(captionResampleWatchdog);
          captionResampleWatchdog = undefined;
        }
        console.info("[AccessPlay Captions] Audio chunk resampling completed", {
          chunk: chunkIndex + 1,
          elapsedMs: Math.round(performance.now() - resampleStarted),
          outputSamples: samples.length
        });
      }
      if (captionSkipToChunkIndex !== null && captionSkipToChunkIndex > chunkIndex) {
        chunkIndex = captionSkipToChunkIndex;
        captionSkipToChunkIndex = null;
        continue;
      }
      const inferenceComplete = new Promise((resolve) => {
        captionInferenceResolver = resolve;
      });
      captionWorker.postMessage({
        type: "transcribe",
        audio: samples,
        offsetSeconds,
        progress: (chunkIndex + 1) / totalChunks,
        finalChunk: chunkIndex + 1 === totalChunks
      }, [samples.buffer]);
      setCaptionLoadingStatus("Loading...", generation);
      await inferenceComplete;
      chunkIndex++;
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    if (isCurrentCaptionGeneration(generation) && !captionComplete) {
      captionWorker.postMessage({ type: "finish" });
    }
  }).catch((error) => {
    if (!isCurrentCaptionGeneration(generation)) return;
    if (error.name === "AbortError") return;
    failCaptionLoad(error.message);
  });
}

function scheduleCaptionTranscription() {
  cancelAnimationFrame(captionStartFrame);
  captionStartFrame = requestAnimationFrame(() => {
    captionStartFrame = requestAnimationFrame(() => {
      captionStartFrame = undefined;
      startCaptionTranscription();
    });
  });
}

function updateCaptionCue() {
  if (!showCaptions.checked) return false;
  const cue = captionChunks.find((chunk) => audio.currentTime >= chunk.start && audio.currentTime <= chunk.end);
  const text = cue?.text || "";
  setCaptionDisplayText(text);
  return Boolean(cue?.text);
}

showCaptions.addEventListener("change", () => {
  captionControlTouched = true;
  if (showCaptions.checked) {
    setCaptionDisplayText("");
    scheduleCaptionTranscription();
    updateCaptionCue();
  } else {
    cancelCaptionTranscription();
  }
});

// --- 2. Helper Functions ---
function formatTime(seconds) {
  if (isNaN(seconds)) return "0:00";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) {
    return `${h}:${m < 10 ? "0" : ""}${m}:${s < 10 ? "0" : ""}${s}`;
  }
  return `${m}:${s < 10 ? "0" : ""}${s}`;
}

function formatSpeechTime(seconds) {
  if (isNaN(seconds)) return "0 seconds";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  let res = "";
  if (h > 0) res += h + " hour" + (h === 1 ? "" : "s") + " ";
  if (m > 0) res += m + " minute" + (m === 1 ? "" : "s") + " ";
  res += s + " second" + (s === 1 ? "" : "s");
  return res.trim();
}

function updateVolume(val) {
  const newVol = Math.min(1, Math.max(0, val));
  audio.volume = newVol;
  volumeSlider.value = Math.round(newVol * 100);
  announceLive(newVol === 0 ? "Muted" : `Volume ${volumeSlider.value}%`);
}

function updateSpeed(val) {
  const newSpeed = Math.min(5, Math.max(0.5, val));
  audio.playbackRate = newSpeed;
  speedSlider.value = newSpeed.toFixed(1);
  announceLive(`Speed ${speedSlider.value}X`);
}

function announceAndClear(msg) {
  announceLive(msg);
}

function announceLive(message) {
  const alreadyDisplayed = ariaLive.textContent === message;
  clearTimeout(liveAnnounceTimer);
  clearTimeout(liveUpdateTimer);
  clearTimeout(liveHideTimer);
  ariaLive.hidden = false;
  liveAnnounceTimer = setTimeout(() => {
    if (alreadyDisplayed) {
      ariaLive.textContent = "";
    liveUpdateTimer = setTimeout(() => {
      ariaLive.textContent = message;
      liveHideTimer = setTimeout(() => {
        ariaLive.hidden = true;
        }, 200);
      }, 100);
    } else {
      ariaLive.textContent = message;
      liveHideTimer = setTimeout(() => {
        ariaLive.hidden = true;
      }, 200);
    }
  }, 50);
}

// --- 3. Seek Dialog Logic ---
const seekDialog = document.getElementById("seekDialog");
const seekHours = document.getElementById("seekHours");
const seekMinutes = document.getElementById("seekMinutes");
const seekSeconds = document.getElementById("seekSeconds");
const seekGo = document.getElementById("seekGo");
const seekCancel = document.getElementById("seekCancel");
const seekPause = document.getElementById("seekPause");
const seekLive = document.getElementById("seekLive");

timeContainer.onclick = () => {
  const cur = audio.currentTime;
  seekHours.value = Math.floor(cur / 3600);
  seekMinutes.value = Math.floor((cur % 3600) / 60);
  seekSeconds.value = Math.floor(cur % 60);
  seekPause.textContent = audio.paused ? "Play" : "Pause";
  seekLive.textContent = "";
  seekDialog.showModal();
};

seekPause.onclick = () => {
  if (audio.paused) {
    audio.play();
    seekPause.textContent = "Pause";
    announceLive("Playing");
  } else {
    audio.pause();
    seekPause.textContent = "Play";
    announceLive("Paused");
  }
};

seekGo.onclick = () => {
  let h = parseInt(seekHours.value);
  let m = parseInt(seekMinutes.value);
  let s = parseInt(seekSeconds.value);
  if (isNaN(h)) h = 0;
  if (isNaN(m)) m = 0;
  if (isNaN(s)) s = 0;

  const totalSeconds = (h * 3600) + (m * 60) + s;

  if (totalSeconds > audio.duration) {
    seekLive.textContent = "Error: Seek time exceeds duration";
    setTimeout(() => { seekLive.textContent = ""; }, 2000);
  } else {
    audio.currentTime = totalSeconds;
    seekDialog.close();
  }
};

seekCancel.onclick = () => {
  seekDialog.close();
};

excludeDomain.onclick = () => {
  if (document.body.hasAttribute("data-accessplay-demo")) {
    const original = document.getElementById("originalMedia");
    if (!original) return;
    const wasPlaying = !audio.paused;
    original.src = audio.currentSrc || src;
    original.currentTime = Number.isFinite(audio.currentTime) ? audio.currentTime : 0;
    original.volume = audio.volume;
    original.playbackRate = audio.playbackRate;
    original.hidden = false;
    document.body.dataset.accessplayDemoExcluded = "true";
    audio.pause();
    if (wasPlaying) original.play().catch(() => {});
    return;
  }
  let sourceHostname = "";
  try { sourceHostname = new URL(src || window.location.href).hostname; } catch {}
  window.parent.postMessage({ accessPlayAction: "exclude-this-domain", sourceHostname }, "*");
};

// --- 4. Event Handlers ---
playBtn.onclick = () => {
  if (audio.paused) {
    if (showCaptions.checked) {
      if (captionSettingsLoaded) scheduleCaptionTranscription();
      else playPressedBeforeCaptionSettings = true;
    }
    audio.play();
    announceLive("Playing");
  } else {
    audio.pause();
    announceLive("Paused");
  }
};

rewindBtn.onclick = () => {
  audio.currentTime = Math.max(0, audio.currentTime - skipSeconds);
  announceLive(`Rewind ${formatSpeechTime(skipSeconds)}`);
};

forwardBtn.onclick = () => {
  if (!isNaN(audio.duration)) {
    audio.currentTime = Math.min(audio.duration, audio.currentTime + skipSeconds);
    announceLive(`Fast forward ${formatSpeechTime(skipSeconds)}`);
  }
};

downloadBtn.onclick = () => {
  if (src) {
    chrome.runtime.sendMessage({ download: src });
  }
};

volumeSlider.oninput = () => {
  updateVolume(volumeSlider.value / 100);
};

speedSlider.oninput = () => {
  updateSpeed(parseFloat(speedSlider.value));
};

// Keyboard Shortcuts - FULL RESTORE
window.addEventListener("keydown", (e) => {
  const target = document.activeElement;
  const tagName = target.tagName;
  const typingInput = tagName === "INPUT" && !["range", "checkbox", "button"].includes(target.type);
  if (e.ctrlKey || e.altKey || e.metaKey || target.isContentEditable ||
      tagName === "TEXTAREA" || tagName === "SELECT" || typingInput) {
    return;
  }
  if (e.key === " " && (tagName === "BUTTON" || (tagName === "INPUT" && target.type === "checkbox"))) return;
  if (tagName === "INPUT" && target.type === "range" && e.key.startsWith("Arrow")) return;

  switch (e.key) {
    case "c":
    case "C":
      e.preventDefault();
      showCaptions.click();
      break;
    case " ":
      e.preventDefault();
      playBtn.click();
      break;
    case "ArrowLeft":
      if (document.activeElement.id !== "volume" && document.activeElement.id !== "speed") {
        e.preventDefault();
        rewindBtn.click();
      }
      break;
    case "ArrowRight":
      if (document.activeElement.id !== "volume" && document.activeElement.id !== "speed") {
        e.preventDefault();
        forwardBtn.click();
      }
      break;
    case "ArrowUp":
      e.preventDefault();
      updateVolume(audio.volume + 0.05);
      break;
    case "ArrowDown":
      e.preventDefault();
      updateVolume(audio.volume - 0.05);
      break;
    case "[":
      e.preventDefault();
      if (audio.playbackRate <= 0.5) {
        announceAndClear("Already minimum speed");
      } else {
        updateSpeed(audio.playbackRate - 0.1);
      }
      break;
    case "]":
      e.preventDefault();
      if (audio.playbackRate >= 5.0) {
        announceAndClear("Already maximum speed");
      } else {
        updateSpeed(audio.playbackRate + 0.1);
      }
      break;
    case "=":
      e.preventDefault();
      if (audio.playbackRate === 1.0) {
        announceAndClear("Already normal speed");
      } else {
        updateSpeed(1);
      }
      break;
    case "+":
      e.preventDefault();
      if (audio.playbackRate >= 5.0) {
        announceAndClear("Already maximum speed");
      } else {
        updateSpeed(5);
      }
      break;
    case "-":
      e.preventDefault();
      if (audio.playbackRate <= 0.5) {
        announceAndClear("Already minimum speed");
      } else {
        updateSpeed(0.5);
      }
      break;
  }
});

// --- 5. State Syncing ---
audio.addEventListener("loadedmetadata", () => {
  durationTimeSpan.textContent = formatTime(audio.duration);
});

audio.addEventListener("timeupdate", () => {
  const cur = audio.currentTime;
  const dur = audio.duration;
  currentTimeSpan.textContent = formatTime(cur);
  timeContainer.setAttribute("aria-label", `${formatSpeechTime(cur)} of ${formatSpeechTime(dur)}`);
  timeContainer.setAttribute("aria-braillelabel", `${formatTime(cur)}/${formatTime(dur)}`);
  updateCaptionCue();
});

audio.addEventListener("seeking", () => {
  if (!showCaptions.checked || !captionLoadStarted || captionComplete) return;
  const hasCaptionAtCurrentPosition = captionChunks.some(
    (chunk) => audio.currentTime >= chunk.start && audio.currentTime <= chunk.end
  );
  if (!hasCaptionAtCurrentPosition) {
    const nextChunk = Math.max(0, Math.floor((audio.currentTime - captionTranscriptionStartOffset) / 30));
    captionSkipToChunkIndex = Math.max(captionSkipToChunkIndex ?? 0, nextChunk);
  }
});

audio.addEventListener("play", () => {
  playBtn.textContent = "Pause";
  seekPause.textContent = "Pause";
});

audio.addEventListener("pause", () => {
  playBtn.textContent = "Play";
  seekPause.textContent = "Play";
});

window.addEventListener("pagehide", () => {
  captionWorker?.terminate();
  if (!globalThis.caches) return;
  const modelPath = `/${captionModel}/resolve/`;
  void caches.open("transformers-cache").then(async (cache) => {
    const requests = await cache.keys();
    await Promise.all(requests
      .filter((request) => new URL(request.url).pathname.includes(modelPath))
      .map((request) => cache.delete(request)));
  }).catch((error) => console.warn("AccessPlay demo could not clear the caption model cache", error));
});
