import { app } from "../../scripts/app.js";

const TIME_GATE_CLASS = "ApplyTimeGatedLoRAToModelLTX23";
const SEGMENT_PLANNER_CLASS = "LTXVSemanticSegmentPlanner";
const SCHEDULE_SYNC_CLASS = "LTXVScheduleSync";
const SCHEDULED_TIME_GATE_CLASS = "LTXVTimeGatedLoRAScheduled";

const REGION_WIDGETS = [
  "effect_region",
  "envelope_mode",
  "strength_before",
  "strength_during",
  "strength_after",
  "ramp_mode",
  "q_in",
  "q_out",
];
const MANUAL_WIDGETS = ["segment_lengths", "segment_strengths"];
const MEMORY_WIDGETS = ["memory_mode"];

const PLANNER_BOUNDARY_WIDGETS = ["boundary_1", "boundary_2", "boundary_3", "boundary_4", "boundary_5"];
const DIRECTOR_BOUNDARY_WIDGETS = ["boundary_1", "boundary_2", "boundary_3"];
const MAX_DIRECTOR_SEGMENTS = 4;

function findWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function asInt(value, fallback = 0) {
  const parsed = parseInt(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function countPromptSegments(text) {
  const raw = String(text ?? "").replace(/\r\n?/g, "\n").trim();
  if (!raw) return 0;

  if (raw.includes("|")) {
    return raw.split("|").map((part) => part.trim()).filter(Boolean).length;
  }

  const collectMarkers = (regex) => {
    const found = [];
    let match;
    while ((match = regex.exec(raw)) !== null) {
      found.push({ index: match.index, number: Number(match[1]) });
    }
    return found;
  };

  let markers = collectMarkers(/^[ \t]*(?:\[\s*)?(?:segment|scene|shot)\s*[-_#]?\s*(\d+)\s*(?:\]\s*)?(?:[:.\-–—])?[ \t]*/gim);
  if (markers.length < 2) {
    markers = collectMarkers(/(?:\[\s*)?(?:segment|scene|shot)\s*[-_#]?\s*(\d+)\s*(?:\]\s*)?\s*[:\-–—]\s*/gi);
  }
  if (markers.length >= 2 && raw.slice(0, markers[0].index).trim() === "") {
    const ordered = markers.every((marker, idx) => marker.number === idx + 1);
    if (ordered) return markers.length;
  }

  const paragraphs = raw.split(/(?:\n[ \t]*){2,}/).map((part) => part.trim()).filter(Boolean);
  if (paragraphs.length >= 2 && paragraphs.length <= MAX_DIRECTOR_SEGMENTS) {
    return paragraphs.length;
  }
  return 1;
}

function setInactive(widget, inactive, message) {
  if (!widget) return;
  widget.options ??= {};
  widget.disabled = inactive;
  widget.options.disabled = inactive;
  widget.__ltxInactive = inactive;
  widget.__ltxInactiveMessage = inactive ? message : "";
  if (widget.__ltxBaseLabel === undefined) {
    widget.__ltxBaseLabel = widget.label ?? widget.name;
  }
  widget.label = inactive ? `${widget.__ltxBaseLabel} [inactive]` : widget.__ltxBaseLabel;
}

function refreshTimeGateWidgets(node) {
  const scheduleMode = findWidget(node, "schedule_mode")?.value ?? "effect_region";
  const envelopeMode = findWidget(node, "envelope_mode")?.value ?? "local";
  const rampMode = findWidget(node, "ramp_mode")?.value ?? "q_curve";

  const isRegionMode = scheduleMode === "effect_region";
  const isManualMode = scheduleMode === "manual_frames";
  const isV12Envelope = isRegionMode && (envelopeMode === "local" || envelopeMode === "hold_strength");
  const usesQ = isV12Envelope && rampMode === "q_curve";
  const usesTransitionFrames = isManualMode || (isRegionMode && (envelopeMode === "transition_frames" || envelopeMode === "v1.0rc1_transition_frames"));

  for (const name of REGION_WIDGETS) {
    setInactive(findWidget(node, name), !isRegionMode, "Inactive in manual_frames mode.");
  }
  for (const name of MANUAL_WIDGETS) {
    setInactive(findWidget(node, name), !isManualMode, "Inactive in effect_region mode.");
  }
  for (const name of MEMORY_WIDGETS) {
    setInactive(findWidget(node, name), false, "");
  }

  setInactive(findWidget(node, "envelope_mode"), !isRegionMode, "Inactive in manual_frames mode.");
  setInactive(findWidget(node, "ramp_mode"), !isV12Envelope, "Only active in local/hold_strength envelope mode.");
  setInactive(findWidget(node, "q_in"), !usesQ, "Only active when ramp_mode is q_curve in local/hold_strength envelope mode.");
  setInactive(findWidget(node, "q_out"), !usesQ, "Only active when ramp_mode is q_curve in local/hold_strength envelope mode.");
  setInactive(findWidget(node, "transition_frames"), !usesTransitionFrames, "Ignored by local/hold_strength envelopes; the neighboring segment length defines the q-curve ramp duration.");

  dirtyCanvas(node);
}

function refreshPlannerWidgets(node) {
  const semanticSource = findWidget(node, "semantic_source")?.value ?? "local_prompts";
  const manualCount = Math.max(1, Math.min(6, asInt(findWidget(node, "segment_count")?.value, 3)));
  const promptCount = Math.max(1, Math.min(6, countPromptSegments(findWidget(node, "local_prompts")?.value)));
  const segmentCount = semanticSource === "local_prompts" ? promptCount : manualCount;

  setInactive(findWidget(node, "local_prompts"), semanticSource !== "local_prompts", "Only used when semantic_source=local_prompts.");
  setInactive(findWidget(node, "segment_count"), semanticSource !== "manual_count", "Only used when semantic_source=manual_count.");

  PLANNER_BOUNDARY_WIDGETS.forEach((name, idx) => {
    const active = idx < Math.max(0, segmentCount - 1);
    setInactive(findWidget(node, name), !active, `Active for segment_count >= ${idx + 2}.`);
  });

  node.setDirtyCanvas?.(true, true);
}

function wrapWidgetCallback(node, name, refreshFn) {
  const widget = findWidget(node, name);
  if (!widget || widget.__ltxCallbackWrapped) return;
  widget.__ltxCallbackWrapped = true;
  const originalCallback = widget.callback;
  widget.callback = function (...args) {
    const result = originalCallback?.apply(this, args);
    refreshFn(node);
    return result;
  };
}

function wrapTimeGateCallbacks(node) {
  for (const name of ["schedule_mode", "envelope_mode", "ramp_mode"]) {
    wrapWidgetCallback(node, name, refreshTimeGateWidgets);
  }
}

function refreshScheduledTimeGateWidgets(node) {
  const rampMode = findWidget(node, "ramp_mode")?.value ?? "q_curve";
  const usesQ = rampMode === "q_curve";
  for (const name of ["q_in", "q_out", "q_in_shift", "q_out_shift"]) {
    setInactive(findWidget(node, name), !usesQ, "Only active when ramp_mode=q_curve. Shift controls also require relay_timing / PromptRelay epsilon softness.");
  }
  dirtyCanvas(node);
}

function wrapScheduledTimeGateCallbacks(node) {
  wrapWidgetCallback(node, "ramp_mode", refreshScheduledTimeGateWidgets);
}

function wrapPlannerCallbacks(node) {
  for (const name of ["semantic_source", "segment_count", "local_prompts"]) {
    wrapWidgetCallback(node, name, refreshPlannerWidgets);
  }
}

function clamp01(value) {
  return Math.max(0.001, Math.min(0.999, Number(value) || 0));
}

function directorSegmentCount(node) {
  const count = countPromptSegments(findWidget(node, "local_prompts")?.value);
  return Math.max(1, Math.min(MAX_DIRECTOR_SEGMENTS, count || 1));
}

function directorBoundaryValues(node, segmentCount) {
  const needed = Math.max(0, Math.min(3, segmentCount - 1));
  const mode = findWidget(node, "boundary_mode")?.value ?? "equal";
  const values = [];
  for (let i = 0; i < needed; i++) {
    if (mode === "equal") {
      values.push((i + 1) / segmentCount);
    } else {
      values.push(clamp01(findWidget(node, DIRECTOR_BOUNDARY_WIDGETS[i])?.value ?? ((i + 1) / segmentCount)));
    }
  }
  values.sort((a, b) => a - b);
  return values;
}

function setDirectorBoundary(node, index, value) {
  const widget = findWidget(node, DIRECTOR_BOUNDARY_WIDGETS[index]);
  if (!widget) return;
  const segmentCount = directorSegmentCount(node);
  const needed = Math.max(0, Math.min(3, segmentCount - 1));
  if (index >= needed) return;

  const minGap = 0.015;
  const values = directorBoundaryValues(node, segmentCount);
  let v = clamp01(value);
  const prev = index > 0 ? values[index - 1] + minGap : minGap;
  const next = index < needed - 1 ? values[index + 1] - minGap : 1.0 - minGap;
  v = Math.max(prev, Math.min(next, v));

  const modeWidget = findWidget(node, "boundary_mode");
  if (modeWidget && modeWidget.value !== "custom") {
    modeWidget.value = "custom";
    modeWidget.callback?.(modeWidget.value);
  }
  widget.value = Number(v.toFixed(3));
  widget.callback?.(widget.value);
  node.setDirtyCanvas?.(true, true);
}


function dirtyCanvas(node) {
  node?.setDirtyCanvas?.(true, true);
  app?.canvas?.setDirty?.(true, true);
}

function localPointerPos(node, event, pos) {
  if (Array.isArray(pos) && pos.length >= 2) {
    return [Number(pos[0]) || 0, Number(pos[1]) || 0];
  }
  if (event && Number.isFinite(event.canvasX) && Number.isFinite(event.canvasY)) {
    return [event.canvasX - node.pos[0], event.canvasY - node.pos[1]];
  }
  const canvas = app?.canvas;
  if (canvas?.convertEventToCanvasOffset && event) {
    try {
      const cpos = canvas.convertEventToCanvasOffset(event);
      if (Array.isArray(cpos) && cpos.length >= 2) {
        return [cpos[0] - node.pos[0], cpos[1] - node.pos[1]];
      }
    } catch (err) {
      // Fall through to offset coordinates.
    }
  }
  return [Number(event?.offsetX) || 0, Number(event?.offsetY) || 0];
}

function handleScheduleBoundaryPointer(node, event, pos, phase) {
  const rect = node.__ltxBoundaryBarRect;
  if (!rect) return false;

  const needed = Math.max(0, Math.min(3, directorSegmentCount(node) - 1));
  if (!needed) return false;

  const [lx, ly] = localPointerPos(node, event, pos);
  const insideX = lx >= rect.x && lx <= rect.x + rect.w;
  const insideY = ly >= rect.y - 16 && ly <= rect.y + rect.h + 16;

  const values = directorBoundaryValues(node, rect.segmentCount);
  let nearest = -1;
  let nearestDist = Infinity;
  for (let i = 0; i < needed; i++) {
    const boundaryValue = values[i] ?? ((i + 1) / (needed + 1));
    const hx = rect.x + boundaryValue * rect.w;
    const d = Math.abs(lx - hx);
    if (d < nearestDist) {
      nearest = i;
      nearestDist = d;
    }
  }

  if (phase === "down") {
    // Click anywhere on the bar: grab the nearest boundary handle and move it to the click.
    // This is more reliable in Comfy/LiteGraph than depending on custom-widget mouse callbacks.
    if (insideX && insideY && nearest >= 0) {
      node.__ltxDraggingBoundary = nearest;
      setDirectorBoundary(node, nearest, (lx - rect.x) / rect.w);
      event?.preventDefault?.();
      event?.stopPropagation?.();
      dirtyCanvas(node);
      return true;
    }
  }

  if (phase === "move") {
    if (Number.isInteger(node.__ltxDraggingBoundary)) {
      setDirectorBoundary(node, node.__ltxDraggingBoundary, (lx - rect.x) / rect.w);
      event?.preventDefault?.();
      event?.stopPropagation?.();
      dirtyCanvas(node);
      return true;
    }
  }

  if (phase === "up" || phase === "leave") {
    if (Number.isInteger(node.__ltxDraggingBoundary)) {
      setDirectorBoundary(node, node.__ltxDraggingBoundary, (lx - rect.x) / rect.w);
      node.__ltxDraggingBoundary = null;
      event?.preventDefault?.();
      event?.stopPropagation?.();
      dirtyCanvas(node);
      return true;
    }
  }

  return false;
}

function installScheduleBoundaryNodeHandlers(nodeType) {
  if (nodeType.prototype.__ltxBoundaryNodeHandlersInstalled) return;
  nodeType.prototype.__ltxBoundaryNodeHandlersInstalled = true;

  const originalOnMouseDown = nodeType.prototype.onMouseDown;
  nodeType.prototype.onMouseDown = function (event, pos, canvas) {
    if (handleScheduleBoundaryPointer(this, event, pos, "down")) return true;
    return originalOnMouseDown?.apply(this, arguments);
  };

  const originalOnMouseMove = nodeType.prototype.onMouseMove;
  nodeType.prototype.onMouseMove = function (event, pos, canvas) {
    if (handleScheduleBoundaryPointer(this, event, pos, "move")) return true;
    return originalOnMouseMove?.apply(this, arguments);
  };

  const originalOnMouseUp = nodeType.prototype.onMouseUp;
  nodeType.prototype.onMouseUp = function (event, pos, canvas) {
    if (handleScheduleBoundaryPointer(this, event, pos, "up")) return true;
    return originalOnMouseUp?.apply(this, arguments);
  };

  const originalOnMouseLeave = nodeType.prototype.onMouseLeave;
  nodeType.prototype.onMouseLeave = function (event, pos, canvas) {
    if (handleScheduleBoundaryPointer(this, event, pos, "leave")) return true;
    return originalOnMouseLeave?.apply(this, arguments);
  };
}

function addScheduleBoundaryBar(node) {
  if (node.__ltxScheduleBoundaryBarAdded) return;
  node.__ltxScheduleBoundaryBarAdded = true;
  const widget = {
    name: "schedule_boundary_bar",
    type: "ltxv_boundary_bar",
    serialize: false,
    computeSize(width) {
      return [width, 88];
    },
    draw(ctx, node, width, y, height) {
      const segmentCount = directorSegmentCount(node);
      const values = directorBoundaryValues(node, segmentCount);
      const x = 12;
      const barY = y + 22;
      const barW = Math.max(120, width - 24);
      const barH = 38;
      const colors = ["#3b6fa8", "#b66b34", "#4d8d4d", "#8959a8"];
      const edges = [0, ...values, 1];

      node.__ltxBoundaryBarRect = { x, y: barY, w: barW, h: barH, values, segmentCount };
      ctx.save();
      ctx.font = "11px sans-serif";
      ctx.textBaseline = "top";
      ctx.fillStyle = "#9aa4b2";
      ctx.fillText(`Schedule boundary bar · ${segmentCount} segment${segmentCount === 1 ? "" : "s"}`, x, y + 4);

      for (let i = 0; i < segmentCount; i++) {
        const x0 = x + edges[i] * barW;
        const x1 = x + edges[i + 1] * barW;
        ctx.fillStyle = colors[i % colors.length];
        ctx.globalAlpha = 0.82;
        ctx.fillRect(x0, barY, Math.max(1, x1 - x0), barH);
        ctx.globalAlpha = 1.0;
        ctx.strokeStyle = "rgba(230,235,245,0.65)";
        ctx.strokeRect(x0, barY, Math.max(1, x1 - x0), barH);
        ctx.fillStyle = "#f4f7fb";
        ctx.fillText(`S${i + 1}`, x0 + 5, barY + 5);
        const pct = Math.round((edges[i + 1] - edges[i]) * 100);
        ctx.fillStyle = "#dbe4ef";
        ctx.fillText(`${pct}%`, x0 + 5, barY + 21);
      }

      for (let i = 0; i < values.length; i++) {
        const hx = x + values[i] * barW;
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(hx, barY - 5);
        ctx.lineTo(hx, barY + barH + 5);
        ctx.stroke();
        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(hx, barY - 7, 5, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    },
    mouse(event, pos, node) {
      const rect = node.__ltxBoundaryBarRect;
      if (!rect) return false;
      const type = String(event?.type || "").toLowerCase();
      const lx = Array.isArray(pos) ? pos[0] : (event?.canvasX ?? 0) - node.pos[0];
      const ly = Array.isArray(pos) ? pos[1] : (event?.canvasY ?? 0) - node.pos[1];
      const needed = Math.max(0, Math.min(3, directorSegmentCount(node) - 1));
      if (!needed) return false;

      if (type.includes("down")) {
        let nearest = -1;
        let nearestDist = Infinity;
        for (let i = 0; i < needed; i++) {
          const hx = rect.x + (rect.values[i] ?? ((i + 1) / (needed + 1))) * rect.w;
          const d = Math.abs(lx - hx);
          if (d < nearestDist) {
            nearest = i;
            nearestDist = d;
          }
        }
        const insideY = ly >= rect.y - 14 && ly <= rect.y + rect.h + 14;
        if (nearest >= 0 && nearestDist <= 18 && insideY) {
          node.__ltxDraggingBoundary = nearest;
          return true;
        }
      }
      if (type.includes("move") || type.includes("drag")) {
        if (Number.isInteger(node.__ltxDraggingBoundary)) {
          const v = (lx - rect.x) / rect.w;
          setDirectorBoundary(node, node.__ltxDraggingBoundary, v);
          return true;
        }
      }
      if (type.includes("up") || type.includes("leave") || type.includes("cancel")) {
        if (Number.isInteger(node.__ltxDraggingBoundary)) {
          node.__ltxDraggingBoundary = null;
          return true;
        }
      }
      return false;
    },
  };
  node.addCustomWidget?.(widget);
}

function refreshScheduleSyncWidgets(node) {
  const count = directorSegmentCount(node);
  const promptCount = countPromptSegments(findWidget(node, "local_prompts")?.value);
  const boundaryMode = findWidget(node, "boundary_mode")?.value ?? "equal";
  DIRECTOR_BOUNDARY_WIDGETS.forEach((name, idx) => {
    const active = boundaryMode === "custom" && idx < Math.max(0, count - 1);
    setInactive(findWidget(node, name), !active, active ? "" : "Use boundary_mode=custom and at least two segments.");
  });
  const expected = findWidget(node, "expected_segments");
  if (expected && !["auto", "1", "2", "3", "4"].includes(String(expected.value))) {
    expected.value = "auto";
  }
  if (promptCount > MAX_DIRECTOR_SEGMENTS) {
    node.bgcolor = "#4a2222";
  }
  node.setDirtyCanvas?.(true, true);
}

function wrapScheduleSyncCallbacks(node) {
  for (const name of ["local_prompts", "boundary_mode", ...DIRECTOR_BOUNDARY_WIDGETS]) {
    wrapWidgetCallback(node, name, refreshScheduleSyncWidgets);
  }
}

app.registerExtension({
  name: "LTXV.TimeGatedLoRA.UI.v1.3.0",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === TIME_GATE_CLASS) {
      const originalOnCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = originalOnCreated?.apply(this, arguments);
        wrapTimeGateCallbacks(this);
        setTimeout(() => refreshTimeGateWidgets(this), 0);
        return result;
      };

      const originalOnConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const result = originalOnConfigure?.apply(this, arguments);
        wrapTimeGateCallbacks(this);
        setTimeout(() => refreshTimeGateWidgets(this), 0);
        return result;
      };

      const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;
      nodeType.prototype.onConnectionsChange = function () {
        const result = originalOnConnectionsChange?.apply(this, arguments);
        setTimeout(() => refreshTimeGateWidgets(this), 0);
        return result;
      };
    }

    if (nodeData.name === SCHEDULED_TIME_GATE_CLASS) {
      const originalOnCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = originalOnCreated?.apply(this, arguments);
        wrapScheduledTimeGateCallbacks(this);
        setTimeout(() => refreshScheduledTimeGateWidgets(this), 0);
        return result;
      };

      const originalOnConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const result = originalOnConfigure?.apply(this, arguments);
        wrapScheduledTimeGateCallbacks(this);
        setTimeout(() => refreshScheduledTimeGateWidgets(this), 0);
        return result;
      };

      const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;
      nodeType.prototype.onConnectionsChange = function () {
        const result = originalOnConnectionsChange?.apply(this, arguments);
        setTimeout(() => refreshScheduledTimeGateWidgets(this), 0);
        return result;
      };
    }

    if (nodeData.name === SEGMENT_PLANNER_CLASS) {
      const originalOnCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = originalOnCreated?.apply(this, arguments);
        wrapPlannerCallbacks(this);
        setTimeout(() => refreshPlannerWidgets(this), 0);
        return result;
      };

      const originalOnConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const result = originalOnConfigure?.apply(this, arguments);
        wrapPlannerCallbacks(this);
        setTimeout(() => refreshPlannerWidgets(this), 0);
        return result;
      };
    }


    if (nodeData.name === SCHEDULE_SYNC_CLASS) {
      installScheduleBoundaryNodeHandlers(nodeType);
      const originalOnCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = originalOnCreated?.apply(this, arguments);
        addScheduleBoundaryBar(this);
        wrapScheduleSyncCallbacks(this);
        setTimeout(() => refreshScheduleSyncWidgets(this), 0);
        return result;
      };

      const originalOnConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const result = originalOnConfigure?.apply(this, arguments);
        addScheduleBoundaryBar(this);
        wrapScheduleSyncCallbacks(this);
        setTimeout(() => refreshScheduleSyncWidgets(this), 0);
        return result;
      };
    }
  },
});
