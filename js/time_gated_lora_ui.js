import { app } from "../../scripts/app.js";

const NODE_CLASS = "ApplyTimeGatedLoRAToModelLTX23";
const REGION_WIDGETS = ["effect_region", "envelope_mode", "strength_before", "strength_during", "strength_after", "ramp_mode", "ramp_q"];
const MANUAL_WIDGETS = ["segment_lengths", "segment_strengths"];

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
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
    widget.label = inactive ? `${widget.__ltxBaseLabel}  [inactive]` : widget.__ltxBaseLabel;
}

function refreshWidgets(node) {
    const scheduleMode = findWidget(node, "schedule_mode")?.value ?? "effect_region";
    const envelopeMode = findWidget(node, "envelope_mode")?.value ?? "local";
    const rampMode = findWidget(node, "ramp_mode")?.value ?? "flat";
    const isRegionMode = scheduleMode === "effect_region";
    const isManualMode = scheduleMode === "manual_frames";
    const isV11Envelope = isRegionMode && (envelopeMode === "local" || envelopeMode === "hold_strength");
    const usesQ = isV11Envelope && rampMode === "q_curve";
    const usesTransitionFrames = isManualMode || (isRegionMode && (envelopeMode === "transition_frames" || envelopeMode === "v1.0rc1_transition_frames"));

    for (const name of REGION_WIDGETS) {
        setInactive(findWidget(node, name), !isRegionMode, "Inactive in manual_frames mode.");
    }
    for (const name of MANUAL_WIDGETS) {
        setInactive(findWidget(node, name), !isManualMode, "Inactive in effect_region mode.");
    }

    setInactive(findWidget(node, "ramp_mode"), !isV11Envelope, "Inactive in manual_frames and transition_frames mode.");
    setInactive(findWidget(node, "ramp_q"), !usesQ, usesQ ? "" : "Only active when ramp_mode is q_curve in local/hold_strength envelope mode.");
    setInactive(findWidget(node, "transition_frames"), !usesTransitionFrames, "Ignored by v1.1 local/hold_strength envelopes; segment length defines the ramp duration.");

    node.setDirtyCanvas?.(true, true);
}

function wrapWidgetCallback(node, name) {
    const widget = findWidget(node, name);
    if (!widget || widget.__ltxCallbackWrapped) return;
    widget.__ltxCallbackWrapped = true;
    const originalCallback = widget.callback;
    widget.callback = function (...args) {
        const result = originalCallback?.apply(this, args);
        refreshWidgets(node);
        return result;
    };
}

function wrapCallbacks(node) {
    for (const name of ["schedule_mode", "envelope_mode", "ramp_mode"]) {
        wrapWidgetCallback(node, name);
    }
}

app.registerExtension({
    name: "LTXV.TimeGatedLoRA.UI.v1.1rc9",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const originalOnCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnCreated?.apply(this, arguments);
            wrapCallbacks(this);
            setTimeout(() => refreshWidgets(this), 0);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            wrapCallbacks(this);
            setTimeout(() => refreshWidgets(this), 0);
            return result;
        };

        const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = originalOnConnectionsChange?.apply(this, arguments);
            setTimeout(() => refreshWidgets(this), 0);
            return result;
        };
    },
});
