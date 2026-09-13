/**
 * navigation.js
 *
 * Updates the "Target" panel UI from server navigation payloads and speaks
 * guidance using the browser's speechSynthesis API (Section 21, 28, 45).
 * Every spoken message is also shown visually, as required.
 */

(function () {
    "use strict";

    const targetEmpty = document.getElementById("target-empty");
    const targetInfo = document.getElementById("target-info");
    if (!targetInfo) return; // Not on the camera page.

    const targetName = document.getElementById("target-name");
    const targetDirection = document.getElementById("target-direction");
    const targetDistance = document.getElementById("target-distance");
    const targetStatus = document.getElementById("target-status");
    const targetConfidence = document.getElementById("target-confidence");
    const transcriptEl = document.getElementById("voice-transcript");

    function prettyName(name) {
        if (!name) return "";
        return name.replace(/_/g, " ");
    }

    function speak(text) {
        if (!text) return;
        transcriptEl.innerHTML = `<span>🔊</span> <span>AI Guidance: <strong>"${text}"</strong></span>`;
        if (!window.speechSynthesis) return;
        // Cancel any queued utterance so guidance doesn't stack up and lag
        // behind real-time movement.
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        window.speechSynthesis.speak(utterance);
    }

    function showTargetPanel() {
        targetEmpty.classList.add("hidden");
        targetInfo.classList.remove("hidden");
        targetInfo.classList.add("active-target");
    }

    function hideTargetPanel() {
        targetEmpty.classList.remove("hidden");
        targetInfo.classList.add("hidden");
        targetInfo.classList.remove("active-target");
    }

    function handleUpdate(nav) {
        if (!nav || !nav.target_active) {
            return;
        }

        showTargetPanel();
        targetName.textContent = prettyName(nav.target_class);

        if (nav.found) {
            targetDirection.textContent = (nav.direction || "-").toUpperCase();
            targetDistance.textContent = nav.distance_label || "unavailable";
            targetConfidence.textContent = nav.confidence != null ? Math.round(nav.confidence * 100) + "%" : "-";
            targetStatus.textContent = describeStatus(nav);
        } else if (nav.grace_period) {
            targetStatus.textContent = "Briefly out of view...";
        } else if (nav.lost) {
            targetStatus.textContent = "Target lost";
            targetDirection.textContent = "-";
        } else {
            targetStatus.textContent = "Searching...";
        }

        if (nav.speak_text) {
            speak(nav.speak_text);
        }
    }

    function describeStatus(nav) {
        if (nav.low_confidence) return "Low confidence";
        if (nav.multiple_detected) return `Getting Closer (${nav.detected_count} detected)`;
        return "Tracking";
    }

    function handleControlMessage(data) {
        if (data.type === "target_set") {
            showTargetPanel();
            targetName.textContent = prettyName(data.object);
            targetStatus.textContent = "Searching...";
            targetDirection.textContent = "-";
            targetDistance.textContent = "-";
            targetConfidence.textContent = "-";
        } else if (data.type === "target_cleared") {
            hideTargetPanel();
        } else if (data.type === "voice_response") {
            if (data.object) {
                showTargetPanel();
                targetName.textContent = prettyName(data.object);
                targetStatus.textContent = "Searching...";
            }
            if (data.intent === "stop_tracking") {
                hideTargetPanel();
            }
            speak(data.speak_text);
        }
    }

    window.AIVisionNavigation = { handleUpdate, handleControlMessage };
})();
