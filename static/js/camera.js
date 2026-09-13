/**
 * camera.js
 *
 * Handles WebRTC camera capture and frame sampling (Section 4, 5, 34).
 * The <video> element displays the full-rate camera feed smoothly (~24-30
 * FPS, handled natively by the browser). We separately sample frames at
 * the configured AI FPS (4/6/8) onto a hidden canvas, JPEG-encode them,
 * and stream them to the Django Channels WebSocket for AI processing.
 *
 * No frame is ever stored: each sampled frame is encoded, sent, and
 * discarded immediately (Section 6, 35).
 */

(function () {
    "use strict";

    const root = document.getElementById("app-root");
    if (!root) return; // Not on the camera page.

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = root.dataset.wsUrl ? root.dataset.wsUrl.replace(/^wss?:/, wsProtocol) : `${wsProtocol}//${window.location.host}/ws/vision/`;
    const cameraWidth = parseInt(root.dataset.cameraWidth, 10) || 640;
    const cameraHeight = parseInt(root.dataset.cameraHeight, 10) || 480;
    let aiFps = parseInt(root.dataset.defaultAiFps, 10) || 6;

    const video = document.getElementById("webcam");
    const placeholder = document.getElementById("camera-placeholder");
    const btnCapture = document.getElementById("btn-capture");
    const btnStop = document.getElementById("btn-stop");
    const btnMic = document.getElementById("btn-mic");
    const fpsSelect = document.getElementById("ai-fps-select");

    const statusCamera = document.getElementById("status-camera");
    const statusAi = document.getElementById("status-ai");
    const statusTracking = document.getElementById("status-tracking");

    const devCamera = document.getElementById("dev-camera");
    const devAi = document.getElementById("dev-ai");
    const devModel = document.getElementById("dev-model");
    const devFps = document.getElementById("dev-fps");
    const devInference = document.getElementById("dev-inference");
    const devDepth = document.getElementById("dev-depth");

    // Hidden sampling canvas (never displayed, never persisted to disk).
    const sampleCanvas = document.createElement("canvas");
    sampleCanvas.width = cameraWidth;
    sampleCanvas.height = cameraHeight;
    const sampleCtx = sampleCanvas.getContext("2d", { willReadFrequently: true });

    let mediaStream = null;
    let ws = null;
    let sampleIntervalId = null;
    let awaitingResponse = false; // simple back-pressure guard

    window.AIVisionApp = window.AIVisionApp || {};
    const App = window.AIVisionApp;
    App.currentTargetClass = null;

    function setDot(el, state) {
        el.classList.remove("dot-off", "dot-active", "dot-processing");
        el.classList.add(state === "active" ? "dot-active" : state === "processing" ? "dot-processing" : "dot-off");
    }

    async function startCamera() {
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: cameraWidth }, height: { ideal: cameraHeight } },
                audio: false,
            });
        } catch (err) {
            if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
                alert("Camera access was denied. Please allow camera access in your browser settings.");
            } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
                alert("No camera detected. Please connect a webcam.");
            } else {
                alert("Could not access the camera: " + err.message);
            }
            return;
        }

        video.srcObject = mediaStream;
        placeholder.classList.add("hidden");
        setDot(statusCamera, "active");
        devCamera.textContent = "Active";

        mediaStream.getVideoTracks()[0].addEventListener("ended", () => {
            alert("Camera connection lost.");
            stopCamera();
        });

        btnCapture.disabled = true;
        btnStop.disabled = false;
        btnMic.disabled = false;

        connectWebSocket();
    }

    function stopCamera() {
        if (sampleIntervalId) {
            clearInterval(sampleIntervalId);
            sampleIntervalId = null;
        }
        if (mediaStream) {
            mediaStream.getTracks().forEach((track) => track.stop());
            mediaStream = null;
        }
        video.srcObject = null;
        placeholder.classList.remove("hidden");

        if (ws) {
            ws.close();
            ws = null;
        }

        setDot(statusCamera, "off");
        setDot(statusAi, "off");
        setDot(statusTracking, "off");
        devCamera.textContent = "Inactive";
        devAi.textContent = "Inactive";

        btnCapture.disabled = false;
        btnStop.disabled = true;
        btnMic.disabled = true;

        if (window.AIVisionOverlay) {
            window.AIVisionOverlay.clear();
        }
    }

    function connectWebSocket() {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            setDot(statusAi, "active");
            devAi.textContent = "Active";
            startSampling();
        };

        ws.onmessage = (event) => {
            let data;
            try {
                data = JSON.parse(event.data);
            } catch (e) {
                return;
            }
            handleServerMessage(data);
        };

        ws.onerror = () => {
            console.error("WebSocket error");
        };

        ws.onclose = () => {
            setDot(statusAi, "off");
            devAi.textContent = "Inactive";
        };
    }

    function startSampling() {
        const intervalMs = Math.round(1000 / aiFps);
        if (sampleIntervalId) clearInterval(sampleIntervalId);
        sampleIntervalId = setInterval(sampleAndSendFrame, intervalMs);
    }

    function sampleAndSendFrame() {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        if (awaitingResponse) return; // avoid piling up frames if the server is still busy
        if (video.readyState < 2) return;

        sampleCtx.drawImage(video, 0, 0, sampleCanvas.width, sampleCanvas.height);
        const dataUrl = sampleCanvas.toDataURL("image/jpeg", 0.85);

        awaitingResponse = true;
        setDot(statusAi, "processing");

        ws.send(JSON.stringify({ type: "frame", image: dataUrl }));
    }

    function handleServerMessage(data) {
        if (data.type === "status") {
            devModel.textContent = (data.model && data.model.model_name) || "unavailable";
            devFps.textContent = data.ai_fps;
            return;
        }

        if (data.type === "error") {
            console.warn("Server error:", data.message);
            awaitingResponse = false;
            setDot(statusAi, "active");
            return;
        }

        if (data.type === "detections") {
            awaitingResponse = false;
            setDot(statusAi, "active");

            devFps.textContent = data.fps;
            devInference.textContent = data.inference_ms + " ms";
            devDepth.textContent = data.depth_ran_this_frame ? (data.depth_ms + " ms") : "skipped (cached)";

            if (window.AIVisionOverlay) {
                window.AIVisionOverlay.drawDetections(data.objects, video.videoWidth || sampleCanvas.width,
                    video.videoHeight || sampleCanvas.height);
            }

            if (window.AIVisionNavigation) {
                window.AIVisionNavigation.handleUpdate(data.navigation);
            }

            setDot(statusTracking, data.navigation && data.navigation.target_active ? "active" : "off");
            return;
        }

        if (data.type === "target_set" || data.type === "target_cleared" || data.type === "voice_response" ||
            data.type === "ai_fps_set") {
            if (window.AIVisionNavigation) {
                window.AIVisionNavigation.handleControlMessage(data);
            }
        }
    }

    // --- UI wiring ---
    btnCapture.addEventListener("click", startCamera);
    btnStop.addEventListener("click", stopCamera);

    fpsSelect.addEventListener("change", () => {
        aiFps = parseInt(fpsSelect.value, 10);
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "set_ai_fps", fps: aiFps }));
        }
        if (sampleIntervalId) startSampling();
    });

    // Keyboard shortcuts (Section 49)
    document.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

        if (e.code === "Space") {
            e.preventDefault();
            if (mediaStream) stopCamera(); else startCamera();
        } else if (e.key.toLowerCase() === "v" && !btnMic.disabled) {
            btnMic.click();
        } else if (e.key === "Escape") {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: "clear_target" }));
            }
        }
    });

    App.sendWsMessage = function (message) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(message));
        }
    };

    App.isConnected = function () {
        return !!(ws && ws.readyState === WebSocket.OPEN);
    };
})();
