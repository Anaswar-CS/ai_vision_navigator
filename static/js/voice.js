/**
 * voice.js
 *
 * Browser-native Speech Recognition for voice input (Section 19).
 * Text-to-speech output lives in navigation.js (speaks server responses),
 * since that's where the navigation/voice_response messages arrive.
 */

(function () {
    "use strict";

    const btnMic = document.getElementById("btn-mic");
    const statusVoice = document.getElementById("status-voice");
    const transcriptEl = document.getElementById("voice-transcript");
    const textForm = document.getElementById("text-command-form");
    const textInput = document.getElementById("text-command");

    if (!btnMic) return; // Not on the camera page.

    const SpeechRecognitionImpl = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition = null;
    let listening = false;   // true only after onstart fires
    let started = false;     // true from start() call until onend fires

    function setDot(el, state) {
        el.classList.remove("dot-off", "dot-active", "dot-processing");
        el.classList.add(state === "active" ? "dot-active" : state === "processing" ? "dot-processing" : "dot-off");
    }

    function sendVoiceCommand(text) {
        if (window.AIVisionApp && window.AIVisionApp.sendWsMessage) {
            window.AIVisionApp.sendWsMessage({ type: "voice_command", text: text });
        }
    }

    function buildRecognition() {
        if (!SpeechRecognitionImpl) {
            console.warn("[VOICE] SpeechRecognition is NOT available in this browser window.");
            return null;
        }
        console.log("[VOICE] Instantiating new SpeechRecognition object...");
        const r = new SpeechRecognitionImpl();
        r.lang = "en-US";
        r.continuous = false;
        r.interimResults = false;

        r.onstart = () => {
            console.log("[VOICE] onstart fired: Speech recognition session is ACTIVE and listening.");
            listening = true;
            setDot(statusVoice, "processing");
            btnMic.classList.add("listening");
            transcriptEl.classList.add("speaking");
            transcriptEl.innerHTML = '<span>🎙️</span> <span>Listening... Speak your command now.</span>';
        };

        r.onaudiostart = () => {
            console.log("[VOICE] onaudiostart: Audio capture has started.");
        };

        r.onspeechstart = () => {
            console.log("[VOICE] onspeechstart: User speech detected by browser.");
        };

        r.onresult = (event) => {
            console.log("[VOICE] onresult fired:", event);
            if (event.results && event.results.length > 0 && event.results[0].length > 0) {
                const text = event.results[0][0].transcript;
                console.log("[VOICE] Recognized text:", text);
                transcriptEl.innerHTML = `<span>💬</span> <span>You said: <strong>"${text}"</strong></span>`;
                sendVoiceCommand(text);
            }
        };

        r.onerror = (event) => {
            console.error("[VOICE] onerror fired:", event.error, event);
            btnMic.classList.remove("listening");
            transcriptEl.classList.remove("speaking");
            if (event.error === "aborted") return;
            listening = false;
            started = false;
            setDot(statusVoice, "active");
            // Rebuild so next click gets a completely fresh object
            recognition = buildRecognition();

            if (event.error === "network") {
                transcriptEl.innerHTML = '<span>⚠️</span> <span>Voice speech recognition was blocked (Brave Shields or no network). Try <strong>Chrome / Edge</strong> or type below.</span>';
            } else if (event.error === "not-allowed" || event.error === "service-not-allowed") {
                transcriptEl.innerHTML = '<span>⚠️</span> <span>Microphone access blocked. Please allow mic permissions in browser settings.</span>';
            } else if (event.error === "no-speech") {
                transcriptEl.innerHTML = '<span>⚠️</span> <span>No speech detected. Please speak closer to the microphone.</span>';
            } else if (event.error === "audio-capture") {
                transcriptEl.innerHTML = '<span>⚠️</span> <span>No microphone found or audio capture failed.</span>';
            } else {
                transcriptEl.innerHTML = '<span>⚠️</span> <span>Voice recognition error (' + event.error + '). Type your command below.</span>';
            }
        };

        r.onend = () => {
            console.log("[VOICE] onend fired: Speech recognition session ended.");
            listening = false;
            started = false;
            btnMic.classList.remove("listening");
            transcriptEl.classList.remove("speaking");
            setDot(statusVoice, "active");
        };

        return r;
    }

    if (SpeechRecognitionImpl) {
        recognition = buildRecognition();
        setDot(statusVoice, "active");
    } else {
        console.warn("[VOICE] SpeechRecognitionImpl is null.");
        btnMic.title = "Speech recognition is not supported in this browser.";
    }

    btnMic.addEventListener("click", () => {
        console.log("[VOICE] 'Ask AI Voice' button clicked. Current state: started=" + started + ", listening=" + listening);
        if (!recognition) {
            console.error("[VOICE] Cannot start: recognition object is null.");
            alert("Speech recognition is not supported in this browser. Try Chrome or Edge, "
                + "or use the text command box instead.");
            return;
        }
        if (started) {
            console.log("[VOICE] Stopping active recognition session...");
            try { recognition.stop(); } catch (err) {
                console.warn("[VOICE] Error stopping recognition:", err);
            }
            started = false;
            listening = false;
            setDot(statusVoice, "active");
        } else {
            console.log("[VOICE] Calling recognition.start()...");
            try {
                started = true;
                recognition.start();
                console.log("[VOICE] recognition.start() called successfully without synchronous exception.");
            } catch (err) {
                console.error("[VOICE] Synchronous exception in recognition.start():", err);
                started = false;
                listening = false;
                setDot(statusVoice, "active");
                transcriptEl.textContent = "Could not start voice recognition. Please try again.";
                recognition = buildRecognition();
            }
        }
    });

    if (textForm) {
        textForm.addEventListener("submit", (e) => {
            e.preventDefault();
            const text = textInput.value.trim();
            if (!text) return;
            transcriptEl.innerHTML = `<span>💬</span> <span>You typed: <strong>"${text}"</strong></span>`;
            sendVoiceCommand(text);
            textInput.value = "";
        });
    }
})();
