(() => {
  let currentRunId = null;
  let currentManifest = null;
  let activeVariantId = null;
  let pollTimer = null;
  let activeTargetId = null;
  let magicJobId = null;
  let magicSession = null;
  let magicVariant = "enhanced";
  let magicPalette = "";
  let standalone = false;
  let standalonePath = "";
  let lastRenderedId = null;
  let lastRenderedStrength = null;
  let magicBusy = false;
  let activePage = "run";

  const pages = {
    run: document.getElementById("page-run"),
    gallery: document.getElementById("page-gallery"),
    magic: document.getElementById("page-magic"),
    edit: document.getElementById("page-edit"),
  };
  const navLinks = {
    run: document.getElementById("nav-run"),
    gallery: document.getElementById("nav-gallery"),
    magic: document.getElementById("nav-magic"),
    edit: document.getElementById("nav-edit"),
  };

  const runForm = document.getElementById("run-form");
  const fileForm = document.getElementById("file-form");
  const inputPathEl = document.getElementById("input-path");
  const outputPathEl = document.getElementById("output-path");
  const initialPaletteEl = document.getElementById("initial-palette");
  const startRunBtn = document.getElementById("start-run-btn");
  const sourceSummary = document.getElementById("source-summary");
  const activeSourcePill = document.getElementById("active-source-pill");

  const statusPanel = document.getElementById("job-status-panel");
  const statusOpenGalleryBtn = document.getElementById("status-open-gallery");
  const jobBadge = document.getElementById("job-badge");
  const progressBar = document.getElementById("job-progress-bar");
  const stageText = document.getElementById("job-stage-text");
  const errorBox = document.getElementById("job-error-box");
  const warningsBox = document.getElementById("job-warnings-box");

  const workspace = document.getElementById("gallery-workspace");
  const galleryEmptyState = document.getElementById("gallery-empty-state");
  const editWorkspace = document.getElementById("edit-workspace");
  const editEmptyState = document.getElementById("edit-empty-state");

  const targetTitle = document.getElementById("active-target-title");
  const targetStory = document.getElementById("target-story");
  const variantSubtitle = document.getElementById("active-variant-subtitle");
  const activeBadges = document.getElementById("active-badges");
  const mainPreviewImg = document.getElementById("main-preview-img");
  const manualPreviewImg = document.getElementById("manual-preview-img");
  const manualCurrentTitle = document.getElementById("manual-current-title");
  const variantDesc = document.getElementById("active-variant-desc");
  const variantProv = document.getElementById("active-variant-prov");
  const variantCountBadge = document.getElementById("variant-count-badge");
  const downloadTiffBtn = document.getElementById("download-tiff-btn");
  const downloadPngBtn = document.getElementById("download-png-btn");

  window.PhotonloomHistogram.bind(
    mainPreviewImg,
    document.getElementById("display-histogram"),
    document.getElementById("histogram-caption")
  );

  const rerenderPaletteEl = document.getElementById("rerender-palette");
  const applyPresetBtn = document.getElementById("apply-preset-btn");
  const customMixInput = document.getElementById("custom-mix-input");
  const applyMixBtn = document.getElementById("apply-mix-btn");
  const editRequestInput = document.getElementById("edit-request-input");
  const applyEditBtn = document.getElementById("apply-edit-btn");
  const controlFeedback = document.getElementById("control-feedback");
  const finishPresetEl = document.getElementById("finish-preset");
  const trimEdgesEl = document.getElementById("trim-edges");
  const finishCropInfo = document.getElementById("finish-crop-info");
  const applyFinishBtn = document.getElementById("apply-finish-btn");
  const finishStarModeEl = document.getElementById("finish-star-mode");
  const finishStarBlendWrap = document.getElementById("finish-star-blend-wrap");
  const finishStarBlendEl = document.getElementById("finish-star-blend");
  const finishStarBlendVal = document.getElementById("finish-star-blend-value");
  const finishInputs = Object.fromEntries(
    ["exposure", "contrast", "saturation", "gamma", "local_contrast"].map((key) => [
      key,
      document.getElementById(`finish-${key.replaceAll("_", "-")}`),
    ])
  );
  const finishDefaults = new Map();

  const scienceFitsList = document.getElementById("science-fits-list");
  const omittedBox = document.getElementById("omitted-palettes-box");
  const cardsGrid = document.getElementById("variant-cards-grid");

  const magicStart = document.getElementById("magic-start");
  const magicRefresh = document.getElementById("magic-refresh");
  const magicRender = document.getElementById("magic-render");
  const magicStrength = document.getElementById("magic-strength");
  const magicTarget = document.getElementById("magic-target");
  const magicStarlessEl = document.getElementById("magic-starless");
  const magicCards = document.getElementById("magic-cards");
  const magicEmptyState = document.getElementById("magic-empty-state");
  const magicFeedback = document.getElementById("magic-feedback");
  const magicStatus = document.getElementById("magic-status");
  const comparison = document.getElementById("magic-comparison");
  const magicBefore = document.getElementById("magic-before");
  const magicAfter = document.getElementById("magic-after");
  const magicSplit = document.getElementById("magic-split");
  const magicDownloads = document.getElementById("magic-downloads");
  const magicSelectedHeading = document.getElementById("magic-selected-heading");
  const variants = ["natural", "enhanced", "dramatic"];

  function pageFromPathname(pathname) {
    const clean = (pathname || "/").replace(/\/+$/, "") || "/";
    if (clean === "/run") return "run";
    if (clean === "/gallery") return "gallery";
    if (clean === "/magic") return "magic";
    if (clean === "/edit") return "edit";
    return null;
  }

  function hasGallery() {
    return Boolean(!standalone && currentManifest && currentManifest.targets?.length > 0);
  }

  function syncPageEmptyStates() {
    const ready = hasGallery();
    workspace.classList.toggle("hidden", !ready);
    galleryEmptyState.classList.toggle("hidden", ready);
    editWorkspace.classList.toggle("hidden", !ready);
    editEmptyState.classList.toggle("hidden", ready);
    navLinks.gallery.classList.toggle("is-disabled", standalone);
    navLinks.edit.classList.toggle("is-disabled", standalone);
  }

  function navigateToPage(page, pushHistory = true) {
    if (!pages[page]) page = "run";
    activePage = page;
    for (const [key, section] of Object.entries(pages)) {
      section.classList.toggle("hidden", key !== page);
    }
    for (const [key, link] of Object.entries(navLinks)) {
      link.classList.toggle("active", key === page);
      if (key === page) {
        link.setAttribute("aria-current", "page");
      } else {
        link.removeAttribute("aria-current");
      }
    }
    syncPageEmptyStates();
    if (pushHistory && window.location.pathname !== `/${page}`) {
      window.history.pushState({ page }, "", `/${page}`);
    }
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  document.querySelectorAll("[data-page-link]").forEach((el) => {
    el.addEventListener("click", (event) => {
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      const targetPage = el.getAttribute("data-page-link");
      navigateToPage(targetPage, true);
    });
  });

  window.addEventListener("popstate", () => {
    const page = pageFromPathname(window.location.pathname) || (hasGallery() ? "gallery" : "run");
    navigateToPage(page, false);
  });

  function setMagicFeedback(msg, isError = false) {
    magicFeedback.textContent = msg || "";
    magicFeedback.classList.toggle("hidden", !msg);
    magicFeedback.classList.toggle("is-error", Boolean(msg && isError));
  }

  function syncStarBlendVisibility() {
    if (!finishStarModeEl || !finishStarBlendWrap) return;
    finishStarBlendWrap.classList.toggle("hidden", finishStarModeEl.value !== "reduced");
  }

  function setFinishValues(recipe) {
    if (!recipe) return;
    finishPresetEl.value = recipe.preset;
    for (const [key, input] of Object.entries(finishInputs)) {
      input.value = recipe[key];
      document.getElementById(`${input.id}-value`).textContent = Number(input.value).toFixed(2);
    }
    if (finishStarModeEl) {
      finishStarModeEl.value = recipe.star_mode || "standard";
      const blend = recipe.star_blend != null ? Number(recipe.star_blend) : 0.45;
      if (finishStarBlendEl && finishStarBlendVal) {
        finishStarBlendEl.value = recipe.star_mode === "reduced" ? blend : 0.45;
        finishStarBlendVal.textContent = Number(finishStarBlendEl.value).toFixed(2);
      }
      syncStarBlendVisibility();
    }
  }

  function readFinishValues() {
    const starMode = finishStarModeEl ? finishStarModeEl.value : "standard";
    const starBlend =
      starMode === "starless"
        ? 0.0
        : starMode === "reduced" && finishStarBlendEl
          ? Number(finishStarBlendEl.value)
          : null;
    return {
      preset: finishPresetEl.value,
      ...Object.fromEntries(
        Object.entries(finishInputs).map(([key, input]) => [key, Number(input.value)])
      ),
      star_mode: starMode,
      ...(starBlend !== null ? { star_blend: starBlend } : {}),
    };
  }

  function setEditBusy(busy) {
    for (const button of [applyFinishBtn, applyPresetBtn, applyMixBtn, applyEditBtn]) {
      button.disabled = busy;
    }
  }

  async function loadFinishPresets() {
    const response = await fetch("/api/finish-presets");
    if (!response.ok) throw new Error("Finish recipes are unavailable");
    const recipes = await response.json();
    finishPresetEl.replaceChildren();
    for (const recipe of recipes) {
      finishDefaults.set(recipe.preset, recipe);
      const option = document.createElement("option");
      option.value = recipe.preset;
      option.textContent = recipe.preset
        .replaceAll("-", " ")
        .replace(/\b\w/g, (letter) => letter.toUpperCase());
      finishPresetEl.appendChild(option);
    }
    setFinishValues(finishDefaults.get("balanced"));
  }

  finishPresetEl.addEventListener("change", () => {
    setFinishValues(finishDefaults.get(finishPresetEl.value));
    showFeedback("Recipe adjusted. Apply Finish to update the image; science FITS stay unchanged.");
  });
  for (const input of Object.values(finishInputs)) {
    input.addEventListener("input", () => {
      document.getElementById(`${input.id}-value`).textContent = Number(input.value).toFixed(2);
      showFeedback("Unsaved display adjustments — use Apply Finish when ready.");
    });
  }
  if (finishStarModeEl) {
    finishStarModeEl.addEventListener("change", () => {
      syncStarBlendVisibility();
      showFeedback("Star layer mode changed — click Apply Finish to rerender.");
    });
  }
  if (finishStarBlendEl) {
    finishStarBlendEl.addEventListener("input", () => {
      if (finishStarBlendVal) {
        finishStarBlendVal.textContent = Number(finishStarBlendEl.value).toFixed(2);
      }
      showFeedback("Star screen-blend weight adjusted — click Apply Finish to rerender.");
    });
  }

  function showFeedback(msg, isError = false) {
    controlFeedback.textContent = msg;
    controlFeedback.style.color = isError ? "#f85149" : "#3fb950";
  }

  function updateStatusUI(job) {
    const isRunning = job.status === "queued" || job.status === "running";
    const isSucceeded = job.status === "succeeded";
    const hasError = Boolean(job.error);
    const warnings = job.warnings || (job.manifest && job.manifest.warnings) || [];
    statusPanel.classList.toggle(
      "hidden",
      standalone || (!isRunning && !hasError && !isSucceeded && warnings.length === 0)
    );
    statusOpenGalleryBtn.classList.toggle("hidden", !isSucceeded);
    jobBadge.textContent = job.status;
    setEditBusy(!isSucceeded);
    const pct = Math.round((job.progress || 0) * 100);
    progressBar.parentElement.classList.toggle("hidden", !isRunning);
    stageText.classList.toggle("hidden", !isRunning && !hasError);
    progressBar.style.width = `${pct}%`;
    stageText.textContent = `${job.stage || job.status} (${pct}%)`;

    if (job.error) {
      errorBox.textContent = `Error: ${job.error}`;
      errorBox.classList.remove("hidden");
    } else {
      errorBox.classList.add("hidden");
    }

    if (warnings.length > 0) {
      warningsBox.innerHTML =
        "<strong>Warnings:</strong><ul>" +
        warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("") +
        "</ul>";
      warningsBox.classList.remove("hidden");
    } else {
      warningsBox.classList.add("hidden");
    }
  }

  function updateSourceHeader() {
    if (standalone) {
      const name = standalonePath.split("/").pop() || "Standalone file";
      activeSourcePill.textContent = `Standalone · ${name}`;
      sourceSummary.textContent = `Standalone Magic View session for ${standalonePath}`;
      syncPageEmptyStates();
      return;
    }
    if (hasGallery()) {
      const target =
        currentManifest.targets.find((t) => t.target_id === activeTargetId) ||
        currentManifest.targets[0];
      const count = (target.science_integrations || []).length;
      activeSourcePill.textContent = `Gallery · ${target.target_label}`;
      sourceSummary.textContent = `Active gallery: "${target.target_label}" (${count} calibrated science channel${count === 1 ? "" : "s"}, ${(target.variants || []).length} variants).`;
    } else {
      activeSourcePill.textContent = "No source loaded";
      sourceSummary.textContent =
        "Process a folder of raw light/calibration frames into a 32-bit FITS gallery, or open a single FITS/TIFF/PNG image directly in Magic View.";
    }
    syncPageEmptyStates();
  }

  function renderGallery(manifest) {
    currentManifest = manifest;
    if (!manifest || !manifest.targets || manifest.targets.length === 0) {
      syncPageEmptyStates();
      return;
    }
    standalone = false;
    const target =
      manifest.targets.find((t) => t.target_id === activeTargetId) || manifest.targets[0];
    activeTargetId = target.target_id;
    updateSourceHeader();
    populateMagicTargets(manifest);
    setEditBusy(target !== manifest.targets[0]);
    if (target !== manifest.targets[0]) {
      showFeedback("Manual rerender applies to the first target only; use Magic View for this target.");
    }
    targetTitle.textContent = target.target_label;
    targetStory.replaceChildren();
    targetStory.classList.add("hidden");
    if (target.about) {
      let url = null;
      try {
        url = new URL(target.about.wikipedia_url);
      } catch {
        /* No link for invalid context metadata. */
      }
      if (url && url.protocol === "https:" && url.hostname === "en.wikipedia.org") {
        const link = document.createElement("a");
        link.href = url.href;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = `${target.about.name} on Wikipedia`;
        targetStory.append(document.createTextNode(`${target.about.summary} `), link);
        targetStory.classList.remove("hidden");
      }
    }

    if (!activeVariantId || !target.variants.some((v) => v.id === activeVariantId)) {
      activeVariantId =
        target.selected_variant || target.primary_variant || target.variants[0].id;
    }

    selectVariant(activeVariantId);

    scienceFitsList.innerHTML = "";
    for (const sci of target.science_integrations || []) {
      const li = document.createElement("li");
      const url = `/api/runs/${encodeURIComponent(currentRunId)}/gallery/${sci.fits_path}`;
      li.innerHTML = `<strong>${escapeHtml(sci.channel)}:</strong> <a href="${url}" download>${escapeHtml(sci.fits_path)}</a> (${sci.width}×${sci.height}, ${sci.total_exposure_s.toFixed(1)}s, ${sci.accepted_frames.length} accepted)`;
      scienceFitsList.appendChild(li);
    }

    if (target.omitted_variants && target.omitted_variants.length > 0) {
      omittedBox.innerHTML =
        "<strong>Omitted Palettes:</strong><ul>" +
        target.omitted_variants
          .map((o) => `<li><strong>${escapeHtml(o.palette)}:</strong> ${escapeHtml(o.reason)}</li>`)
          .join("") +
        "</ul>";
    } else {
      omittedBox.innerHTML = "";
    }

    syncPageEmptyStates();
  }

  function selectVariant(variantId) {
    if (!currentManifest || !currentManifest.targets.length) return;
    const target =
      currentManifest.targets.find((t) => t.target_id === activeTargetId) ||
      currentManifest.targets[0];
    const variant = target.variants.find((v) => v.id === variantId) || target.variants[0];
    if (!variant) return;

    activeVariantId = variant.id;
    variantSubtitle.textContent = `${variant.title} · Palette: ${variant.palette.toUpperCase()}`;
    variantDesc.textContent = variant.description;
    variantProv.textContent = variant.provenance;
    variantCountBadge.textContent = String(target.variants.length);
    rerenderPaletteEl.replaceChildren();
    for (const v of new Map(
      target.variants
        .filter(
          (v) =>
            v.palette !== "custom" &&
            !v.magic &&
            !v.restore_info &&
            (!v.finish || !v.finish.star_mode || v.finish.star_mode === "standard")
        )
        .map((v) => [v.palette, v])
    ).values()) {
      const option = document.createElement("option");
      option.value = v.palette;
      option.textContent = v.title;
      rerenderPaletteEl.appendChild(option);
    }
    if ([...rerenderPaletteEl.options].some((o) => o.value === variant.palette)) {
      rerenderPaletteEl.value = variant.palette;
    }
    if (variant.finish) {
      setFinishValues(variant.finish);
      trimEdgesEl.checked = Boolean(variant.crop);
    }
    const crop = variant.crop;
    finishCropInfo.textContent = crop
      ? `Display: ${crop.width}×${crop.height}, origin (${crop.x}, ${crop.y}); science canvas: ${crop.original_width}×${crop.original_height}.`
      : "Full display canvas. Linear science FITS are retained separately.";

    const cacheBust = `t=${Date.now()}`;
    const pngUrl = `/api/runs/${encodeURIComponent(currentRunId)}/gallery/${variant.png_path}?${cacheBust}`;
    const tiffUrl = `/api/runs/${encodeURIComponent(currentRunId)}/gallery/${variant.tiff_path}`;

    mainPreviewImg.src = pngUrl;
    if (manualPreviewImg) manualPreviewImg.src = pngUrl;
    if (manualCurrentTitle) manualCurrentTitle.textContent = variant.title;
    downloadTiffBtn.href = tiffUrl;
    downloadPngBtn.href = `/api/runs/${encodeURIComponent(currentRunId)}/gallery/${variant.png_path}`;

    activeBadges.innerHTML = "";
    const bPal = document.createElement("span");
    bPal.className = "badge sci";
    bPal.textContent = `Palette: ${variant.palette}`;
    activeBadges.appendChild(bPal);

    if (variant.magic) {
      const bMag = document.createElement("span");
      bMag.className = "badge ai";
      bMag.textContent = variant.magic.variant
        ? `Magic ${variant.magic.variant}`
        : `Magic · ${variant.magic.recipe_id || "Adaptive"}`;
      activeBadges.appendChild(bMag);
    }
    if (variant.finish && variant.finish.star_mode && variant.finish.star_mode !== "standard") {
      const bStar = document.createElement("span");
      bStar.className = "badge sci";
      bStar.textContent =
        variant.finish.star_mode === "starless"
          ? "Starless Nebula"
          : variant.finish.star_mode === "stars-only"
            ? "Stars Only"
            : `Star-Reduced ${Math.round((variant.finish.star_blend ?? 0.45) * 100)}%`;
      activeBadges.appendChild(bStar);
    }
    if (variant.artistic_interpretation) {
      const bArt = document.createElement("span");
      bArt.className = "badge art";
      bArt.textContent = "Artistic Interpretation";
      activeBadges.appendChild(bArt);
    }
    if (variant.ai_enhanced) {
      const bAi = document.createElement("span");
      bAi.className = "badge ai";
      bAi.textContent = "AI-Enhanced Restoration";
      activeBadges.appendChild(bAi);
    }

    cardsGrid.innerHTML = "";
    for (const v of target.variants) {
      const card = document.createElement("div");
      card.className = `variant-card ${v.id === activeVariantId ? "active" : ""}`;
      card.dataset.variantId = v.id;
      const vPng = `/api/runs/${encodeURIComponent(currentRunId)}/gallery/${v.png_path}?${cacheBust}`;
      card.innerHTML = `
        <img src="${vPng}" alt="${escapeHtml(v.title)}">
        <div class="variant-card-body">
          <div class="variant-card-title">${escapeHtml(v.title)}</div>
          <div class="prov-line">${escapeHtml(v.channels_used.join(" + "))}</div>
        </div>
      `;
      card.addEventListener("click", () => selectVariant(v.id));
      cardsGrid.appendChild(card);
    }
  }

  async function pollJob(runId) {
    try {
      const resp = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
      if (!resp.ok) return;
      const job = await resp.json();
      updateStatusUI(job);
      if (job.status === "succeeded") {
        clearInterval(pollTimer);
        pollTimer = null;
        startRunBtn.disabled = false;
        const galResp = await fetch(`/api/runs/${encodeURIComponent(runId)}/gallery`);
        if (galResp.ok) {
          const manifest = await galResp.json();
          activeVariantId = manifest.targets[0]?.selected_variant || null;
          renderGallery(manifest);
          navigateToPage("gallery", true);
        }
      } else if (job.status === "failed") {
        clearInterval(pollTimer);
        pollTimer = null;
        startRunBtn.disabled = false;
      }
    } catch (err) {
      console.error("Poll error:", err);
    }
  }

  runForm.addEventListener("submit", async (e) => {
    standalone = false;
    magicSession = null;
    magicJobId = null;
    lastRenderedId = null;
    lastRenderedStrength = null;
    magicCards.replaceChildren();
    comparison.classList.add("hidden");
    e.preventDefault();
    const input = inputPathEl.value.trim();
    const output = outputPathEl.value.trim() || null;
    const palette = initialPaletteEl.value.trim() || null;
    if (!input) return;

    startRunBtn.disabled = true;
    showFeedback("");
    try {
      const resp = await fetch("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input,
          output,
          palette,
          magic_view: document.getElementById("run-magic-view").checked,
          finish: document.getElementById("run-starless")?.checked
            ? { ...readFinishValues(), star_mode: "starless", star_blend: 0.0 }
            : readFinishValues(),
          trim_edges: trimEdgesEl.checked,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        statusPanel.classList.remove("hidden");
        errorBox.textContent = data.error || "Run request failed";
        errorBox.classList.remove("hidden");
        startRunBtn.disabled = false;
        return;
      }
      currentRunId = data.id;
      activeVariantId = null;
      updateStatusUI(data);
      clearInterval(pollTimer);
      pollTimer = setInterval(() => pollJob(currentRunId), 350);
      pollJob(currentRunId);
    } catch (err) {
      startRunBtn.disabled = false;
      showFeedback(`Request error: ${err.message}`, true);
    }
  });

  applyFinishBtn.addEventListener("click", async () => {
    if (!currentRunId) return;
    setEditBusy(true);
    showFeedback("Applying the full-resolution display finish from linear FITS...");
    try {
      const response = await fetch(`/api/runs/${encodeURIComponent(currentRunId)}/variants`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ finish: readFinishValues(), trim_edges: trimEdgesEl.checked }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Finish failed");
      activeVariantId = data.variant.id;
      renderGallery(data.manifest);
      showFeedback(
        `Applied '${data.variant.finish.preset}' at full resolution. Linear science FITS unchanged.`
      );
    } catch (error) {
      showFeedback(error.message, true);
    } finally {
      setEditBusy(false);
    }
  });

  applyPresetBtn.addEventListener("click", async () => {
    if (!currentRunId) return;
    const palette = rerenderPaletteEl.value;
    showFeedback(`Rerendering preset '${palette}' from linear FITS...`);
    try {
      const resp = await fetch(`/api/runs/${encodeURIComponent(currentRunId)}/variants`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ palette, mix: null }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        showFeedback(data.error || "Failed to rerender preset", true);
        return;
      }
      activeVariantId = data.variant.id;
      renderGallery(data.manifest);
      showFeedback(`Selected and rerendered '${data.variant.id}' without re-stacking.`);
    } catch (err) {
      showFeedback(err.message, true);
    }
  });

  applyMixBtn.addEventListener("click", async () => {
    if (!currentRunId) return;
    const mix = customMixInput.value.trim();
    if (!mix) {
      showFeedback("Enter a custom mix expression (e.g. Ha:1,Ha:0.7+OIII:0.3,OIII:1)", true);
      return;
    }
    showFeedback(`Rerendering custom mix '${mix}'...`);
    try {
      const resp = await fetch(`/api/runs/${encodeURIComponent(currentRunId)}/variants`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ palette: null, mix }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        showFeedback(data.error || "Custom mix failed", true);
        return;
      }
      activeVariantId = data.variant.id;
      renderGallery(data.manifest);
      showFeedback(`Rendered custom mix '${data.variant.title}' from stored linear FITS.`);
    } catch (err) {
      showFeedback(err.message, true);
    }
  });

  applyEditBtn.addEventListener("click", async () => {
    if (!currentRunId) return;
    const request = editRequestInput.value.trim();
    if (!request) {
      showFeedback("Enter an edit instruction first", true);
      return;
    }
    showFeedback(`Applying edit '${request}'...`);
    try {
      const resp = await fetch(`/api/runs/${encodeURIComponent(currentRunId)}/edit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        showFeedback(data.error || "Edit failed", true);
        return;
      }
      activeVariantId = data.variant.id;
      renderGallery(data.manifest);
      showFeedback(`Created edited variant '${data.variant.id}' without re-stacking.`);
    } catch (err) {
      showFeedback(err.message, true);
    }
  });

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function magicAsset(path) {
    if (standalone) {
      return `/api/magic/sessions/${encodeURIComponent(magicJobId)}/assets/${path.split("/").pop()}`;
    }
    return `/api/runs/${encodeURIComponent(currentRunId)}/gallery/${path
      .split("/")
      .map(encodeURIComponent)
      .join("/")}`;
  }

  function setMagicBusy(busy, message = "") {
    magicBusy = busy;
    for (const control of [magicStart, magicRefresh, magicRender, magicTarget]) {
      control.disabled = busy;
    }
    magicStart.disabled = busy || standalone || !currentRunId;
    document.querySelectorAll("#magic-palettes button").forEach((button) => {
      button.disabled = busy;
    });
    magicStatus.textContent = message || (busy ? "Working…" : magicSession ? "Previews ready" : "Ready");
  }

  async function loadExistingGalleryMagic() {
    if (standalone || !currentRunId || !activeTargetId || !magicPalette) return;
    try {
      const params = new URLSearchParams({
        target_id: activeTargetId,
        palette: magicPalette,
      });
      const response = await fetch(
        `/api/runs/${encodeURIComponent(currentRunId)}/magic?${params.toString()}`
      );
      if (!response.ok) return;
      const data = await response.json();
      if (data && data.session) {
        magicJobId = data.id || null;
        magicSession = data.session;
        const latestStrength =
          magicSession.previews?.find((p) => p.variant === magicVariant && p.strength > 0)?.strength ??
          0.5;
        magicStrength.value = latestStrength;
        document.getElementById("magic-strength-value").textContent = Number(latestStrength).toFixed(2);
        if (magicStarlessEl) {
          magicStarlessEl.checked = Boolean(magicSession.starless);
        }
        magicStatus.textContent = "Loaded saved session";
        displayMagic();
      }
    } catch {
      /* Saved session lookup is opportunistic. */
    }
  }

  function populateMagicTargets(manifest) {
    const wrap = document.getElementById("magic-target-wrap");
    wrap.classList.toggle("hidden", manifest.targets.length <= 1);
    magicTarget.replaceChildren();
    for (const target of manifest.targets) {
      const option = document.createElement("option");
      option.value = target.target_id;
      option.textContent = target.target_label;
      magicTarget.append(option);
    }
    magicTarget.value = activeTargetId;
    const target = manifest.targets.find((t) => t.target_id === activeTargetId);
    const palettes = new Map();
    for (const v of target.variants) {
      if (
        v.palette !== "custom" &&
        !v.magic &&
        !v.restore_info &&
        (!v.finish || !v.finish.star_mode || v.finish.star_mode === "standard") &&
        (!palettes.has(v.palette) || v.id === v.palette)
      ) {
        palettes.set(v.palette, v.title);
      }
    }
    if (!palettes.size) palettes.set(target.primary_variant, target.primary_variant);
    if (!palettes.has(magicPalette)) {
      magicPalette = palettes.has(target.primary_variant)
        ? target.primary_variant
        : palettes.keys().next().value;
    }
    const tabs = document.getElementById("magic-palettes");
    tabs.replaceChildren();
    for (const [palette, title] of palettes) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `btn ${palette === magicPalette ? "selected" : ""}`;
      button.textContent = title;
      button.setAttribute("aria-pressed", String(palette === magicPalette));
      button.addEventListener("click", () => {
        magicPalette = palette;
        magicSession = null;
        magicJobId = null;
        lastRenderedId = null;
        lastRenderedStrength = null;
        comparison.classList.add("hidden");
        magicCards.replaceChildren();
        populateMagicTargets(currentManifest);
        setMagicFeedback("This palette starts an independent analysis from original integrations.");
        displayMagic();
      });
      tabs.append(button);
    }
    magicStart.disabled = magicBusy;
    magicRefresh.disabled = magicBusy || !magicSession;
    if (!magicSession) {
      loadExistingGalleryMagic();
    }
  }

  function sameStrength(saved, selected) {
    return Math.abs(saved - selected) < 0.00001;
  }

  function chosenCandidate() {
    return magicSession?.previews?.find(
      (p) => p.variant === magicVariant && sameStrength(p.strength, Number(magicStrength.value))
    );
  }

  function hasAppliedGain(candidate) {
    const a = candidate?.decision?.actual;
    return Boolean(a && (a.tone > 0 || a.detail > 0 || a.sharpen > 0 || a.color > 0));
  }

  function formatAppliedParams(candidate) {
    const a = candidate?.decision?.actual;
    if (!a || !(a.tone > 0 || a.detail > 0 || a.sharpen > 0 || a.color > 0)) {
      return "No enhancement applied (unmodified baseline)";
    }
    return [
      `Tone +${(a.tone * 100).toFixed(1)}%`,
      `Detail +${(a.detail * 100).toFixed(1)}%`,
      `Sharpen +${(a.sharpen * 100).toFixed(1)}%`,
      `Color +${(a.color * 100).toFixed(1)}%`,
    ].join(" · ");
  }

  function variantScaleFactor(variant) {
    return variant === "natural" ? 0.3 : variant === "enhanced" ? 0.65 : 1.0;
  }

  function replayParameters(candidate) {
    if (!candidate) return null;
    if (candidate.decision.proposal) return candidate.decision.proposal;
    const bounds = candidate.decision.nominal;
    return {
      version: 1,
      recipe_id: candidate.decision.recipe_id,
      variant: candidate.variant,
      strength: candidate.strength,
      operator_scales: Object.fromEntries(
        ["tone", "detail", "sharpen", "color"].map((key) => [key, bounds[key] > 0 ? 1 : 0])
      ),
    };
  }

  function displayMagic() {
    magicCards.replaceChildren();
    const hasSession = Boolean(magicSession);
    magicEmptyState.classList.toggle("hidden", hasSession);
    if (!hasSession) {
      comparison.classList.add("hidden");
      magicStart.textContent = "Make Magic View";
      magicStart.classList.add("primary");
      magicStart.disabled = magicBusy || standalone || !currentRunId;
      magicRefresh.textContent = "Update Previews";
      magicRefresh.classList.remove("primary");
      magicRefresh.disabled = true;
      document.getElementById("magic-export-parameters").disabled = true;
      document.getElementById("magic-parameter-preview").textContent =
        "Select a Magic preview to inspect its replayable JSON proposal.";
      return;
    }
    const strength = Number(magicStrength.value);
    const hasAllForStrength = variants.every((v) =>
      magicSession.previews.some((p) => p.variant === v && sameStrength(p.strength, strength))
    );
    magicStart.textContent = "Re-run Analysis";
    magicStart.classList.toggle("primary", false);
    magicStart.disabled = magicBusy || standalone || !currentRunId;
    magicRefresh.textContent = hasAllForStrength
      ? "Previews Up to Date"
      : `Update Previews (${strength.toFixed(2)})`;
    magicRefresh.classList.toggle("primary", !hasAllForStrength);
    magicRefresh.disabled = magicBusy || hasAllForStrength;
    if (!magicBusy) {
      magicStatus.textContent = hasAllForStrength
        ? "Previews ready"
        : `Strength ${strength.toFixed(2)} not computed`;
    }
    const baseline = magicSession.previews.find((p) => p.variant === "natural" && p.strength === 0);
    if (baseline) magicBefore.src = magicAsset(baseline.preview_path);
    for (const variant of variants) {
      const candidate = magicSession.previews.find(
        (p) => p.variant === variant && sameStrength(p.strength, strength)
      );
      const button = document.createElement("button");
      button.type = "button";
      button.className = `magic-card ${magicVariant === variant ? "selected" : ""}`;
      button.setAttribute("aria-pressed", String(magicVariant === variant));
      const header = document.createElement("div");
      header.className = "magic-card-head";
      const title = document.createElement("strong");
      title.textContent = variant.charAt(0).toUpperCase() + variant.slice(1);
      const badge = document.createElement("span");
      badge.className = "magic-card-badge";
      header.append(title, badge);
      button.append(header);
      if (candidate) {
        const applied = hasAppliedGain(candidate);
        badge.textContent = applied
          ? `Applied · ${Math.round(variantScaleFactor(variant) * strength * 100)}% scale`
          : "Baseline (Unchanged)";
        badge.classList.toggle("is-neutral", !applied);
        const image = document.createElement("img");
        image.src = magicAsset(candidate.preview_path);
        image.alt = `${variant} preview`;
        button.append(image);
        const metrics = document.createElement("div");
        metrics.className = `magic-card-metrics ${applied ? "" : "is-neutral"}`;
        metrics.textContent = formatAppliedParams(candidate);
        button.append(metrics);
        const description = document.createElement("small");
        description.textContent = candidate.decision.explanation;
        button.append(description);
      } else {
        badge.textContent = "Not Computed";
        badge.classList.add("is-pending");
        const empty = document.createElement("div");
        empty.className = "magic-card-empty";
        empty.textContent = `Preview not yet computed for strength ${strength.toFixed(2)}. Click "Update Previews (${strength.toFixed(2)})" above to generate.`;
        button.append(empty);
      }
      button.addEventListener("click", () => {
        magicVariant = variant;
        displayMagic();
      });
      magicCards.append(button);
    }
    const selected = chosenCandidate();
    comparison.classList.toggle("hidden", !selected);
    const variantLabel = magicVariant.charAt(0).toUpperCase() + magicVariant.slice(1);
    if (magicSelectedHeading) {
      magicSelectedHeading.textContent = `${variantLabel} · Strength ${strength.toFixed(2)}`;
    }
    const legend = comparison.querySelector(".compare-legend");
    if (selected) {
      const applied = hasAppliedGain(selected);
      magicAfter.src = magicAsset(selected.preview_path);
      magicAfter.style.clipPath = `inset(0 0 0 ${magicSplit.value}%)`;
      if (legend) {
        legend.textContent = applied
          ? `Left: Unmodified Baseline · Right: ${variantLabel} (${formatAppliedParams(selected)})`
          : "Left & Right: Unmodified Baseline (no enhancement applied at this setting)";
      }
      document.getElementById("magic-explanation").textContent =
        `${formatAppliedParams(selected)}. ${selected.decision.explanation} Measurement confidence: ${Math.round(
          (magicSession.analysis.measurement_confidence || 0) * 100
        )}%.` + (magicSession.assumption ? ` ${magicSession.assumption}` : "");
    } else {
      magicAfter.removeAttribute("src");
    }
    magicRender.disabled = magicBusy || !selected || !magicJobId;
    const selectedStem = selected?.preview_path
      ?.split("/")
      .pop()
      ?.replace(/-preview\.png$/, "");
    const saved =
      magicSession.artifacts?.find(
        (a) => a.id === lastRenderedId && sameStrength(lastRenderedStrength ?? -1, strength)
      ) ||
      magicSession.artifacts?.find(
        (a) =>
          (selectedStem && a.id === selectedStem) ||
          (a.magic &&
            a.magic.variant === magicVariant &&
            a.magic.strength != null &&
            sameStrength(a.magic.strength, strength))
      );
    magicDownloads.classList.toggle("hidden", !saved);
    if (saved) {
      document.getElementById("magic-png").href = magicAsset(saved.png_path);
      document.getElementById("magic-tiff").href = magicAsset(saved.tiff_path);
    }
    document.getElementById("magic-report").textContent = JSON.stringify(
      {
        subject: magicSession.subject,
        planner_warning: magicSession.planner_warning,
        proposal: magicSession.proposal,
        preference_prior: magicSession.preference_prior,
        analysis: magicSession.analysis,
        decision: selected?.decision,
      },
      null,
      2
    );
    document.getElementById("magic-export-parameters").disabled = !selected;
    document.getElementById("magic-parameter-preview").textContent = selected
      ? JSON.stringify(replayParameters(selected), null, 2)
      : "";
    const masks = document.getElementById("magic-masks");
    masks.replaceChildren();
    for (const [name, path] of Object.entries(magicSession.analysis?.masks || {})) {
      if (typeof path !== "string" || !path.endsWith(".png")) continue;
      const link = document.createElement("a");
      link.href = magicAsset(path);
      link.textContent = `${name} mask`;
      link.className = "btn";
      link.target = "_blank";
      link.rel = "noopener";
      masks.append(link);
    }
  }

  async function magicRequest(url, payload) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Magic request failed");
    return data;
  }

  async function awaitMagic(job) {
    magicStatus.textContent = job.stage;
    while (job.status === "queued" || job.status === "running") {
      await new Promise((resolve) => setTimeout(resolve, 500));
      const response = await fetch(`/api/magic/sessions/${encodeURIComponent(job.id)}`);
      if (!response.ok) throw new Error("Magic job status unavailable");
      job = await response.json();
      magicStatus.textContent = job.stage;
    }
    if (job.status !== "succeeded") throw new Error(job.error || "Magic job failed");
    magicJobId = job.id;
    magicSession = job.session;
    if (job.stage === "Three previews ready" && magicSession.proposal) {
      magicVariant = magicSession.proposal.variant;
      magicStrength.value = magicSession.proposal.strength;
      document.getElementById("magic-strength-value").textContent = Number(
        magicStrength.value
      ).toFixed(2);
    }
    setMagicFeedback(
      magicSession.planner_warning || job.stage,
      Boolean(magicSession.planner_warning)
    );
    displayMagic();
    return job;
  }

  async function startMagic(payload, url) {
    navigateToPage("magic", true);
    setMagicBusy(true, "Starting Magic View…");
    setMagicFeedback("");
    try {
      await awaitMagic(await magicRequest(url, payload));
    } catch (error) {
      setMagicFeedback(error.message, true);
    } finally {
      setMagicBusy(false);
      displayMagic();
    }
  }

  function magicOptions() {
    return {
      catalog_online: document.getElementById("magic-catalog-online").checked,
      magic_llm: document.getElementById("magic-llm-url").value.trim() || null,
      magic_learn: document.getElementById("magic-learn").checked,
      starless: Boolean(magicStarlessEl && magicStarlessEl.checked),
    };
  }

  document.getElementById("magic-export-parameters").addEventListener("click", () => {
    const candidate = chosenCandidate();
    if (!candidate) return;
    const blob = new Blob([JSON.stringify(replayParameters(candidate), null, 2) + "\n"], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `magic-${candidate.variant}-parameters.json`;
    link.click();
    URL.revokeObjectURL(url);
  });

  document.getElementById("magic-reset-preferences").addEventListener("click", async () => {
    try {
      await magicRequest("/api/magic/preferences/reset", {});
      document.getElementById("magic-preference-history").textContent =
        "Local Magic preference history reset.";
    } catch (error) {
      setMagicFeedback(error.message, true);
    }
  });

  document.getElementById("magic-learn").addEventListener("change", async (event) => {
    const label = document.getElementById("magic-preference-history");
    if (!event.target.checked) {
      label.textContent = "Preference history disabled.";
      return;
    }
    try {
      const response = await fetch("/api/magic/preferences");
      if (!response.ok) throw new Error("Preference history unavailable");
      const info = await response.json();
      label.textContent = info.preference
        ? `Local preference: ${info.preference.class}, ${info.preference.recipe_id}, ${info.preference.variant}, ${info.preference.strength}.`
        : info.warning || "No saved Magic preference.";
    } catch (error) {
      label.textContent = error.message;
    }
  });

  magicStart.addEventListener("click", () => {
    if (standalone || !currentRunId) return;
    startMagic(
      {
        target_id: activeTargetId,
        palette: magicPalette,
        variant: magicVariant,
        strength: Number(magicStrength.value),
        options: magicOptions(),
      },
      `/api/runs/${encodeURIComponent(currentRunId)}/magic`
    );
  });

  magicRefresh.addEventListener("click", () => {
    if (standalone) {
      fileForm.requestSubmit();
    } else {
      magicStart.click();
    }
  });

  magicStrength.addEventListener("input", () => {
    document.getElementById("magic-strength-value").textContent = Number(
      magicStrength.value
    ).toFixed(2);
    displayMagic();
  });

  magicStrength.addEventListener("change", () => {
    if (!magicBusy && magicSession && !chosenCandidate()) magicRefresh.click();
  });

  if (magicStarlessEl) {
    magicStarlessEl.addEventListener("change", () => {
      if (!magicBusy && magicSession) {
        magicRefresh.click();
      } else {
        displayMagic();
      }
    });
  }
  magicTarget.addEventListener("change", () => {
    activeTargetId = magicTarget.value;
    magicPalette = "";
    magicSession = null;
    magicJobId = null;
    lastRenderedId = null;
    lastRenderedStrength = null;
    comparison.classList.add("hidden");
    magicCards.replaceChildren();
    renderGallery(currentManifest);
  });

  magicSplit.addEventListener("input", () => {
    magicAfter.style.clipPath = `inset(0 0 0 ${magicSplit.value}%)`;
  });

  magicRender.addEventListener("click", async () => {
    if (!magicJobId || !chosenCandidate()) return;
    setMagicBusy(true, "Rendering full-resolution image…");
    try {
      const strength = Number(magicStrength.value);
      const url = standalone
        ? `/api/magic/sessions/${encodeURIComponent(magicJobId)}/render`
        : `/api/runs/${encodeURIComponent(currentRunId)}/magic/render`;
      const body = { variant: magicVariant, strength };
      if (!standalone) body.session_id = magicJobId;
      await awaitMagic(await magicRequest(url, body));
      lastRenderedStrength = strength;
      lastRenderedId = magicSession.artifacts?.at(-1)?.id || null;
      if (document.getElementById("magic-learn").checked) {
        document.getElementById("magic-learn").dispatchEvent(new Event("change"));
      }
      if (!standalone) {
        const response = await fetch(`/api/runs/${encodeURIComponent(currentRunId)}/gallery`);
        if (response.ok) renderGallery(await response.json());
      }
    } catch (error) {
      setMagicFeedback(error.message, true);
    } finally {
      setMagicBusy(false);
      displayMagic();
    }
  });

  fileForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const path = document.getElementById("file-path").value.trim();
    const output = document.getElementById("file-output").value.trim();
    if (!path || !output) {
      navigateToPage("magic", true);
      setMagicFeedback("Image path and output directory are required.", true);
      return;
    }
    standalone = true;
    standalonePath = path;
    magicSession = null;
    magicJobId = null;
    lastRenderedId = null;
    lastRenderedStrength = null;
    const paletteTabs = document.getElementById("magic-palettes");
    paletteTabs.replaceChildren();
    const singleChip = document.createElement("span");
    singleChip.className = "badge sci";
    singleChip.textContent = "Single Image Input";
    paletteTabs.append(singleChip);
    statusPanel.classList.add("hidden");
    workspace.classList.add("hidden");
    magicStart.disabled = true;
    updateSourceHeader();
    startMagic(
      {
        path,
        output,
        domain: document.getElementById("file-domain").value,
        transfer: document.getElementById("file-transfer").value,
        variant: magicVariant,
        strength: Number(magicStrength.value),
        options: magicOptions(),
      },
      "/api/magic/sessions"
    );
  });

  (async function loadInitialRun() {
    const requestedPage = pageFromPathname(window.location.pathname);
    try {
      await loadFinishPresets();
      const resp = await fetch("/api/runs");
      if (resp.ok) {
        const runs = await resp.json();
        if (Array.isArray(runs) && runs.length > 0) {
          const latest = runs[0];
          currentRunId = latest.id;
          if (!inputPathEl.value && latest.input) {
            inputPathEl.value = latest.input;
          }
          if (!outputPathEl.value && latest.output) {
            outputPathEl.value = latest.output;
          }
          updateStatusUI(latest);
          if (latest.status === "succeeded" && latest.manifest) {
            activeVariantId = latest.manifest.targets[0]?.selected_variant || null;
            renderGallery(latest.manifest);
            navigateToPage(requestedPage || "gallery", Boolean(requestedPage));
            return;
          } else if (latest.status === "queued" || latest.status === "running") {
            clearInterval(pollTimer);
            pollTimer = setInterval(() => pollJob(currentRunId), 350);
          }
        }
      }
      updateSourceHeader();
      navigateToPage(requestedPage || "run", Boolean(requestedPage));
    } catch (err) {
      console.error("Initial load error:", err);
      navigateToPage(requestedPage || "run", false);
    }
  })();
})();
