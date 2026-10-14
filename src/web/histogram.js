(() => {
  "use strict";

  const MAX_SAMPLE_SIDE = 2048;
  const CHANNELS = ["--hist-red", "--hist-green", "--hist-blue"];
  const FALLBACK_COLORS = ["#f87171", "#4ade80", "#60a5fa"];
  const DESCRIPTION = "RGB histogram of the sRGB display preview, not scientific FITS data.";

  function bind(image, canvas, caption) {
    if (!image || !canvas || !caption) return;

    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", DESCRIPTION);
    if (caption.id) canvas.setAttribute("aria-describedby", caption.id);

    const graph = canvas.getContext("2d");
    if (!graph) {
      caption.textContent = "Display histogram unavailable: canvas is not supported.";
      return;
    }

    const sample = document.createElement("canvas");
    const sampleContext = sample.getContext("2d", { willReadFrequently: true });
    if (!sampleContext) {
      caption.textContent = "Display histogram unavailable: pixel sampling is not supported.";
      return;
    }

    function clear(message) {
      graph.clearRect(0, 0, canvas.width, canvas.height);
      caption.textContent = message;
    }

    function draw() {
      // The current request must have completed. In particular, never draw the
      // previous decoded bitmap during a change of variant or a delayed load.
      if (!image.complete || !image.naturalWidth || !image.naturalHeight ||
          (image.currentSrc && image.currentSrc !== image.src)) {
        clear("Waiting for the current sRGB display preview; no histogram available.");
        return;
      }

      const requestedSrc = image.src;
      const sourceWidth = image.naturalWidth;
      const sourceHeight = image.naturalHeight;
      const scale = Math.min(1, MAX_SAMPLE_SIDE / Math.max(sourceWidth, sourceHeight));
      const width = Math.max(1, Math.round(sourceWidth * scale));
      const height = Math.max(1, Math.round(sourceHeight * scale));

      try {
        sample.width = width;
        sample.height = height;
        sampleContext.drawImage(image, 0, 0, width, height);
        const pixels = sampleContext.getImageData(0, 0, width, height).data;
        // No asynchronous work intervenes between reading this image and painting
        // its chart. The guard also covers a source changed by drawImage hooks.
        if (image.src !== requestedSrc || !image.complete ||
            (image.currentSrc && image.currentSrc !== requestedSrc)) return;

        const bins = [new Uint32Array(256), new Uint32Array(256), new Uint32Array(256)];
        let counted = 0;
        let peak = 0;
        for (let offset = 0; offset < pixels.length; offset += 4) {
          if (pixels[offset + 3] === 0) continue;
          counted++;
          for (let channel = 0; channel < 3; channel++) {
            const count = ++bins[channel][pixels[offset + channel]];
            if (count > peak) peak = count;
          }
        }
        if (!counted) {
          clear("The current display preview has no visible pixels to count.");
          return;
        }

        const w = canvas.width;
        const h = canvas.height;
        graph.clearRect(0, 0, w, h);
        const styles = getComputedStyle(canvas);
        const muted = styles.getPropertyValue("--muted").trim() || "#9ca3af";
        const border = styles.getPropertyValue("--border").trim() || "#6b7280";
        const left = 30;
        const right = Math.max(left + 1, w - 12);
        const top = 10;
        const bottom = Math.max(top + 1, h - 27);
        graph.lineWidth = 1;
        graph.strokeStyle = border;
        graph.beginPath();
        graph.moveTo(left, top);
        graph.lineTo(left, bottom);
        graph.lineTo(right, bottom);
        graph.stroke();

        const normalizer = Math.log1p(peak);
        graph.lineWidth = 1.7;
        for (let channel = 0; channel < 3; channel++) {
          graph.strokeStyle = styles.getPropertyValue(CHANNELS[channel]).trim() || FALLBACK_COLORS[channel];
          graph.beginPath();
          for (let bin = 0; bin < 256; bin++) {
            const x = left + (bin / 255) * (right - left);
            const y = bottom - (Math.log1p(bins[channel][bin]) / normalizer) * (bottom - top);
            if (bin === 0) graph.moveTo(x, y);
            else graph.lineTo(x, y);
          }
          graph.stroke();
        }
        graph.fillStyle = muted;
        graph.font = "11px sans-serif";
        graph.textBaseline = "top";
        graph.textAlign = "left";
        graph.fillText("0", left, bottom + 5);
        graph.textAlign = "right";
        graph.fillText("255", right, bottom + 5);
        graph.textAlign = "left";
        graph.textBaseline = "bottom";
        graph.fillText("log counts", left + 5, top + 12);

        const sampleInfo = scale < 1
          ? `Downsampled ${sourceWidth}×${sourceHeight} to ${width}×${height} pixels.`
          : `Counted ${width}×${height} pixels.`;
        const transparency = counted < width * height ? " Fully transparent pixels excluded." : "";
        caption.textContent = `${DESCRIPTION} ${sampleInfo}${transparency} RGB 8-bit bins (0–255); heights are log(1 + count), normalized to the largest bin.`;
      } catch (_error) {
        clear("Display histogram unavailable for the current preview (pixels could not be read).");
      }
    }

    image.addEventListener("load", draw);
    image.addEventListener("error", () => {
      if (image.complete && image.naturalWidth && image.naturalHeight) {
        draw();
      } else {
        clear("Display histogram unavailable: the current preview could not be loaded.");
      }
    });
    // A new src invalidates the old chart immediately, before its load event.
    new MutationObserver(() => {
      clear("Waiting for the current sRGB display preview; no histogram available.");
      if (image.complete && image.naturalWidth && image.naturalHeight) draw();
    }).observe(image, { attributes: true, attributeFilter: ["src", "srcset", "sizes"] });

    if (image.complete && image.naturalWidth && image.naturalHeight) draw();
    else clear("Waiting for the current sRGB display preview; no histogram available.");
  }

  window.PhotonloomHistogram = { bind };
})();
