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
        if (!SpeechRecognitionImpl) return null;
        const r = new SpeechRecognitionImpl();
        r.lang = "en-US";
        r.continuous = false;
        r.interimResults = false;

        r.onstart = () => {
            listening = true;
            setDot(statusVoice, "processing");
            btnMic.classList.add("listening");
            transcriptEl.classList.add("speaking");
            transcriptEl.innerHTML = '<span>🎙️</span> <span>Listening... Speak your command now.</span>';
        };

        r.onresult = (event) => {
            if (event.results && event.results.length > 0 && event.results[0].length > 0) {
                const text = event.results[0][0].transcript;
                transcriptEl.innerHTML = `<span>💬</span> <span>You said: <strong>"${text}"</strong></span>`;
                sendVoiceCommand(text);
            }
        };

        r.onerror = (event) => {
            // 'aborted' is a normal internal event — not a user-visible error.
            btnMic.classList.remove("listening");
            transcriptEl.classList.remove("speaking");
            if (event.error === "aborted") return;
            console.error("SpeechRecognition error:", event.error, event);
            listening = false;
            started = false;
            setDot(statusVoice, "active");
            // Rebuild so next click gets a completely fresh object
            recognition = buildRecognition();

            if (event.error === "network") {
                transcriptEl.innerHTML = '<span>⚠️</span> <span>Voice recognition requires a stable internet connection.</span>';
            } else if (event.error === "not-allowed" || event.error === "service-not-allowed") {
                transcriptEl.innerHTML = '<span>⚠️</span> <span>Microphone access denied or service unavailable.</span>';
            } else if (event.error === "no-speech") {
                transcriptEl.innerHTML = '<span>⚠️</span> <span>No speech detected. Please try speaking again.</span>';
            } else if (event.error === "audio-capture") {
                transcriptEl.innerHTML = '<span>⚠️</span> <span>No microphone found or audio capture failed.</span>';
            } else {
                transcriptEl.innerHTML = '<span>⚠️</span> <span>Voice recognition error (' + event.error + ').</span>';
            }
        };

        r.onend = () => {
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
        btnMic.title = "Speech recognition is not supported in this browser.";
    }

    btnMic.addEventListener("click", () => {
        if (!recognition) {
            alert("Speech recognition is not supported in this browser. Try Chrome or Edge, "
                + "or use the text command box instead.");
            return;
        }
        if (started) {
            // Session is live — stop it gracefully
            try { recognition.stop(); } catch (err) {
                console.warn("Error stopping recognition:", err);
            }
            started = false;
            listening = false;
            setDot(statusVoice, "active");
        } else {
            // No session running — start a fresh one
            try {
                started = true;
                recognition.start();
            } catch (err) {
                console.error("Failed to start SpeechRecognition:", err);
                started = false;
                listening = false;
                setDot(statusVoice, "active");
                transcriptEl.textContent = "Could not start voice recognition. Please try again.";
                // Rebuild for next attempt
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
