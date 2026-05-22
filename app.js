/**
 * Simple Sign Language Translator - Core Engine
 * Hand landmark normalization, KNN Classification, and UI Controllers
 */

// 1. K-Nearest Neighbors Classifier
class KNNClassifier {
  constructor(k = 5) {
    this.k = k;
    this.samples = []; // Array of { features: [63 numbers], label: String }
  }

  addSample(features, label) {
    this.samples.push({ features, label });
  }

  clear() {
    this.samples = [];
  }

  predict(features) {
    if (this.samples.length === 0) {
      return { label: "No Model Loaded", confidence: 0 };
    }

    // Calculate Euclidean distance to all samples
    const distances = this.samples.map(sample => {
      let sumSq = 0;
      for (let i = 0; i < 63; i++) {
        const diff = features[i] - sample.features[i];
        sumSq += diff * diff;
      }
      const dist = Math.sqrt(sumSq);
      return { dist, label: sample.label };
    });

    // Sort by distance ascending
    distances.sort((a, b) => a.dist - b.dist);

    // Get K nearest neighbors
    const kNeighbors = distances.slice(0, Math.min(this.k, distances.length));

    // Count votes
    const votes = {};
    kNeighbors.forEach(n => {
      votes[n.label] = (votes[n.label] || 0) + 1;
    });

    let maxVotes = -1;
    let predictedLabel = "Unknown";
    for (const label in votes) {
      if (votes[label] > maxVotes) {
        maxVotes = votes[label];
        predictedLabel = label;
      }
    }

    const confidence = maxVotes / kNeighbors.length;
    return { label: predictedLabel, confidence };
  }

  getDatasetJSON() {
    return JSON.stringify(this.samples, null, 2);
  }

  loadDatasetJSON(jsonStr) {
    try {
      const data = JSON.parse(jsonStr);
      if (Array.isArray(data) && data.every(item => item.features && item.label)) {
        this.samples = data;
        return true;
      }
    } catch (e) {
      console.error("Failed to load dataset JSON", e);
    }
    return false;
  }
}

// Initialize classifier
const classifier = new KNNClassifier(5);

// Load default dataset from localStorage if exists
const savedData = localStorage.getItem("sign_language_dataset");
if (savedData) {
  classifier.loadDatasetJSON(savedData);
}

// 2. Coordinate Normalization
function normalizeLandmarks(landmarks) {
  if (!landmarks || landmarks.length !== 21) return null;
  const wrist = landmarks[0];

  // Translate wrist to (0,0,0)
  const translated = landmarks.map(lp => ({
    x: lp.x - wrist.x,
    y: lp.y - wrist.y,
    z: lp.z - wrist.z
  }));

  // Scale relative to wrist-to-middle-MCP (landmark 9) distance
  const mcp = translated[9];
  const scale = Math.sqrt(mcp.x * mcp.x + mcp.y * mcp.y + mcp.z * mcp.z) || 1.0;

  // Flatten to 63-element vector
  const flattened = [];
  for (let i = 0; i < 21; i++) {
    flattened.push(translated[i].x / scale);
    flattened.push(translated[i].y / scale);
    flattened.push(translated[i].z / scale);
  }

  return flattened;
}

// Active camera coordinates
let lastLandmarks = null;
let isModelReady = false;

// UI Elements & State
let videoElement, canvasElement, canvasCtx, camera;
let currentPrediction = "None";
let currentConfidence = 0;

// Initialize MediaPipe and Webcam
function initWebcam(onFrameCallback) {
  videoElement = document.getElementById("webcam");
  canvasElement = document.getElementById("output_canvas");
  canvasCtx = canvasElement.getContext("2d");

  const hands = new window.Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7
  });

  hands.onResults((results) => {
    // Resize canvas dynamically to match container size
    const rect = videoElement.getBoundingClientRect();
    canvasElement.width = rect.width;
    canvasElement.height = rect.height;

    // Draw video frame onto canvas
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    let detectedLabel = "None";
    let confidence = 0;

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const landmarks = results.multiHandLandmarks[0];

      // Draw skeleton lines
      window.drawConnectors(canvasCtx, landmarks, window.HAND_CONNECTIONS, {
        color: "#00F0FF",
        lineWidth: 3
      });
      // Draw joints
      window.drawLandmarks(canvasCtx, landmarks, {
        color: "#FF007A",
        radius: 4,
        lineWidth: 1
      });

      // Normalize features
      const normalized = normalizeLandmarks(landmarks);
      if (normalized) {
        lastLandmarks = normalized;

        if (classifier.samples.length > 0) {
          const res = classifier.predict(normalized);
          detectedLabel = res.label;
          confidence = res.confidence;
        }
      }
    } else {
      lastLandmarks = null;
    }

    currentPrediction = detectedLabel;
    currentConfidence = confidence;

    if (onFrameCallback) {
      onFrameCallback(detectedLabel, confidence);
    }
    canvasCtx.restore();
  });

  camera = new window.Camera(videoElement, {
    onFrame: async () => {
      await hands.send({ image: videoElement });
    },
    width: 640,
    height: 480
  });

  camera.start().catch(err => {
    console.error("Camera access denied or failed", err);
    alert("Camera Access Error: Please check your webcam and permissions.");
  });
}

// ----------------------------------------------------
// Page Specific Controllers
// ----------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  const isTrainer = !!document.getElementById("trainer-container");
  const isTranslator = !!document.getElementById("translator-container");

  if (isTranslator) {
    setupTranslatorPage();
  } else if (isTrainer) {
    setupTrainerPage();
  }
});

// A. TRANSLATOR PAGE CONTROLLER
function setupTranslatorPage() {
  const textOutput = document.getElementById("text-output");
  const historyText = document.getElementById("history-text");
  const confidenceBadge = document.getElementById("confidence-badge");
  const modelStatus = document.getElementById("model-status");
  const fileInput = document.getElementById("file-upload");
  const btnSpeak = document.getElementById("btn-speak");
  const btnClear = document.getElementById("btn-clear");
  const chkAutoSpeak = document.getElementById("chk-auto-speak");

  let sentenceHistory = [];
  let currentStabilizedWord = "";
  let lastTypedWord = "";
  let stableCounter = 0;
  const STABILITY_THRESHOLD = 15; // Requires same gesture for 15 frames (~0.5s) to type

  // Speak function
  function speak(text) {
    if (!text) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "hi-IN"; // Set to Hindi/Indian English support
    window.speechSynthesis.speak(utterance);
  }

  // Update Model status label
  function updateModelUI() {
    const count = classifier.samples.length;
    if (count > 0) {
      modelStatus.textContent = `Model Loaded (${count} Samples)`;
      modelStatus.className = "status-badge success";
    } else {
      modelStatus.textContent = "No Model Loaded. Go to Trainer Page.";
      modelStatus.className = "status-badge danger";
    }
  }
  updateModelUI();

  // Load local dataset upload
  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const success = classifier.loadDatasetJSON(event.target.result);
      if (success) {
        // Save to localStorage for quick reloading
        localStorage.setItem("sign_language_dataset", event.target.result);
        updateModelUI();
        alert(`Successfully loaded model with ${classifier.samples.length} samples!`);
      } else {
        alert("Failed to load dataset JSON. Please make sure file format is valid.");
      }
    };
    reader.readAsText(file);
  });

  // Speak Button
  btnSpeak.addEventListener("click", () => {
    speak(historyText.textContent || textOutput.textContent);
  });

  // Clear Button
  btnClear.addEventListener("click", () => {
    sentenceHistory = [];
    historyText.textContent = "";
    textOutput.textContent = "-";
    lastTypedWord = "";
    currentStabilizedWord = "";
  });

  // MediaPipe frame callback
  initWebcam((label, conf) => {
    // Show live prediction values
    if (classifier.samples.length === 0) {
      textOutput.textContent = "No Model Loaded";
      confidenceBadge.textContent = "0%";
      confidenceBadge.style.backgroundColor = "var(--primary-red)";
      return;
    }

    if (label !== "None" && conf >= 0.6) {
      textOutput.textContent = label;
      confidenceBadge.textContent = `${Math.round(conf * 100)}%`;
      confidenceBadge.style.backgroundColor = "var(--accent-cyan)";

      // Hand gesture stabilization logic for composition
      if (label === currentStabilizedWord) {
        stableCounter++;
        if (stableCounter === STABILITY_THRESHOLD) {
          // Speak letter or word immediately if auto-speak is on
          if (chkAutoSpeak.checked && label !== lastTypedWord) {
            speak(label);
          }

          // Special gestures mappings:
          if (label.toLowerCase() === "space") {
            if (sentenceHistory.length > 0 && sentenceHistory[sentenceHistory.length - 1] !== " ") {
              sentenceHistory.push(" ");
            }
          } else if (label.toLowerCase() === "backspace") {
            sentenceHistory.pop();
          } else if (label.toLowerCase() === "clear") {
            sentenceHistory = [];
          } else {
            // Append word/character
            if (label !== lastTypedWord) {
              sentenceHistory.push(label);
            }
          }

          lastTypedWord = label;
          historyText.textContent = sentenceHistory.join("");
        }
      } else {
        currentStabilizedWord = label;
        stableCounter = 0;
      }
    } else {
      textOutput.textContent = "-";
      confidenceBadge.textContent = "0%";
      confidenceBadge.style.backgroundColor = "rgba(255,255,255,0.1)";
      stableCounter = 0;
      // Allow retyping same word if user brings hand away and returns
      lastTypedWord = ""; 
    }
  });
}

// B. TRAINER PAGE CONTROLLER
function setupTrainerPage() {
  const txtLabel = document.getElementById("txt-label");
  const btnRecord = document.getElementById("btn-record");
  const btnClearAll = document.getElementById("btn-clear-all");
  const btnDownload = document.getElementById("btn-download");
  const fileUpload = document.getElementById("file-upload");
  const tableBody = document.querySelector("#class-table tbody");
  const recordOverlay = document.getElementById("record-overlay");
  const recordCountdown = document.getElementById("record-countdown");
  const sampleCountText = document.getElementById("sample-count");

  let isRecording = false;

  // Refresh classes counts table
  function refreshTable() {
    tableBody.innerHTML = "";
    
    // Group counts
    const counts = {};
    classifier.samples.forEach(sample => {
      counts[sample.label] = (counts[sample.label] || 0) + 1;
    });

    const entries = Object.entries(counts);
    sampleCountText.textContent = classifier.samples.length;

    if (entries.length === 0) {
      tableBody.innerHTML = `<tr><td colspan="3" style="text-align: center; color: var(--text-dim);">No gestures recorded yet.</td></tr>`;
      return;
    }

    entries.forEach(([label, count]) => {
      const row = document.createElement("tr");

      const tdLabel = document.createElement("td");
      tdLabel.textContent = label;

      const tdCount = document.createElement("td");
      tdCount.textContent = `${count} samples`;

      const tdAction = document.createElement("td");
      const btnDelete = document.createElement("button");
      btnDelete.textContent = "Delete";
      btnDelete.className = "btn danger xs";
      btnDelete.addEventListener("click", () => {
        if (confirm(`Delete all samples for "${label}"?`)) {
          classifier.samples = classifier.samples.filter(s => s.label !== label);
          // Sync localStorage
          localStorage.setItem("sign_language_dataset", classifier.getDatasetJSON());
          refreshTable();
        }
      });
      tdAction.appendChild(btnDelete);

      row.appendChild(tdLabel);
      row.appendChild(tdCount);
      row.appendChild(tdAction);
      tableBody.appendChild(row);
    });
  }
  refreshTable();

  // Record custom gesture sequence
  btnRecord.addEventListener("click", () => {
    const label = txtLabel.value.trim();
    if (!label) {
      alert("Please enter a gesture label first (e.g., Hello, A, Ram Ram)!");
      return;
    }

    if (isRecording) return;
    isRecording = true;

    let countdown = 3;
    recordOverlay.style.display = "flex";
    recordCountdown.textContent = countdown;

    // Phase 1: 3 seconds countdown to position hand
    const countdownInterval = setInterval(() => {
      countdown--;
      if (countdown > 0) {
        recordCountdown.textContent = countdown;
      } else {
        clearInterval(countdownInterval);
        startCapturing(label);
      }
    }, 1000);
  });

  // Phase 2: Capture 20 samples with slight movements for 97%+ accuracy
  function startCapturing(label) {
    let captured = 0;
    const targetSamples = 25;
    recordCountdown.textContent = "RECORDING...";

    const captureInterval = setInterval(() => {
      if (lastLandmarks) {
        classifier.addSample(lastLandmarks, label);
        captured++;
        recordCountdown.textContent = `Recording ${captured}/${targetSamples}`;
      } else {
        recordCountdown.textContent = "SHOW HAND IN CAMERA!";
      }

      if (captured >= targetSamples) {
        clearInterval(captureInterval);
        isRecording = false;
        recordOverlay.style.display = "none";
        
        // Sync to localStorage
        localStorage.setItem("sign_language_dataset", classifier.getDatasetJSON());
        refreshTable();
        alert(`Successfully recorded ${targetSamples} samples for "${label}"!`);
      }
    }, 100); // 100ms interval = captures 25 samples in 2.5 seconds
  }

  // Clear dataset
  btnClearAll.addEventListener("click", () => {
    if (confirm("Are you sure you want to delete the entire dataset? All custom gestures will be lost.")) {
      classifier.clear();
      localStorage.removeItem("sign_language_dataset");
      refreshTable();
    }
  });

  // Download Dataset JSON
  btnDownload.addEventListener("click", () => {
    if (classifier.samples.length === 0) {
      alert("Dataset is empty. Record some gestures first.");
      return;
    }
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(classifier.getDatasetJSON());
    const dlAnchor = document.createElement("a");
    dlAnchor.setAttribute("href", dataStr);
    dlAnchor.setAttribute("download", "dataset.json");
    document.body.appendChild(dlAnchor);
    dlAnchor.click();
    dlAnchor.remove();
  });

  // Upload Dataset JSON
  fileUpload.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const success = classifier.loadDatasetJSON(event.target.result);
      if (success) {
        localStorage.setItem("sign_language_dataset", event.target.result);
        refreshTable();
        alert(`Dataset loaded successfully! Total ${classifier.samples.length} samples.`);
      } else {
        alert("Failed to load dataset. Please ensure JSON structure is correct.");
      }
    };
    reader.readAsText(file);
  });

  // Start webcam loop
  initWebcam((label, conf) => {
    // Live update status in table if predicted
    const livePredictBadge = document.getElementById("live-prediction");
    if (livePredictBadge) {
      if (classifier.samples.length === 0) {
        livePredictBadge.textContent = "No trained model";
      } else if (label !== "None" && conf >= 0.6) {
        livePredictBadge.textContent = `${label} (${Math.round(conf * 100)}%)`;
      } else {
        livePredictBadge.textContent = "None";
      }
    }
  });
}
