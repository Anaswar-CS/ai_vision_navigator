/**
 * overlay.js
 *
 * Draws bounding boxes, labels, confidence, and distance on a canvas that
 * sits directly on top of the <video> element (Section 11, 28).
 */

(function () {
    "use strict";

    const canvas = document.getElementById("overlay");
    if (!canvas) return;

    const video = document.getElementById("webcam");
    const ctx = canvas.getContext("2d");

    function resizeCanvasToVideo() {
        const rect = video.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
    }

    window.addEventListener("resize", resizeCanvasToVideo);
    video.addEventListener("loadedmetadata", resizeCanvasToVideo);

    function clear() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    function drawDetections(objects, sourceWidth, sourceHeight) {
        resizeCanvasToVideo();
        clear();
        if (!objects || !objects.length) return;

        const scaleX = canvas.width / (sourceWidth || canvas.width);
        const scaleY = canvas.height / (sourceHeight || canvas.height);

        objects.forEach((obj) => {
            const [x1, y1, x2, y2] = obj.bbox;
            const bx = x1 * scaleX;
            const by = y1 * scaleY;
            const bw = (x2 - x1) * scaleX;
            const bh = (y2 - y1) * scaleY;

            const color = obj.low_confidence ? "#f5a623" : "#35c47a";
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(bx, by, bw, bh);

            const label = `${prettyName(obj.class)} ${(obj.confidence * 100).toFixed(0)}%  ${obj.distance_label || ""}`;
            ctx.font = "13px Segoe UI, sans-serif";
            const textWidth = ctx.measureText(label).width;

            ctx.fillStyle = color;
            ctx.fillRect(bx, Math.max(0, by - 20), textWidth + 10, 20);
            ctx.fillStyle = "#0f1115";
            ctx.fillText(label, bx + 5, Math.max(14, by - 6));
        });
    }

    function prettyName(name) {
        if (!name) return "object";
        return name.replace(/_/g, " ");
    }

    window.AIVisionOverlay = { drawDetections, clear };
})();
