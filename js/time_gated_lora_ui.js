import { app } from "../../scripts/app.js";

const NODE_CLASS = "ApplyTimeGatedLoRAToModelLTX23";
const REGION_WIDGETS = ["effect_region", "strength_before", "strength_during", "strength_after"];
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
    const isRegionMode = scheduleMode === "effect_region";

    for (const name of REGION_WIDGETS) {
        setInactive(findWidget(node, name), !isRegionMode, "Inactive in manual_frames mode.");
    }
    for (const name of MANUAL_WIDGETS) {
        setInactive(findWidget(node, name), isRegionMode, "Inactive in effect_region mode.");
    }


    node.setDirtyCanvas?.(true, true);
}

function wrapScheduleModeCallback(node) {
    const modeWidget = findWidget(node, "schedule_mode");
    if (!modeWidget || modeWidget.__ltxCallbackWrapped) return;
    modeWidget.__ltxCallbackWrapped = true;
    const originalCallback = modeWidget.callback;
    modeWidget.callback = function (...args) {
        const result = originalCallback?.apply(this, args);
        refreshWidgets(node);
        return result;
    };
}

app.registerExtension({
    name: "LTXV.TimeGatedLoRA.UI.v1.0rc1",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const originalOnCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnCreated?.apply(this, arguments);
            wrapScheduleModeCallback(this);
            setTimeout(() => refreshWidgets(this), 0);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            wrapScheduleModeCallback(this);
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
