import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

import {
    formatBatchFlags,
    getBatchSourceRoute,
    MAX_BATCH_SELECTION_ITEMS,
    moveBatchSelection,
    normalizeBatchFlags,
    normalizeBatchSelection,
} from "./batch_selector_utils.mjs";

const BATCH_NODE_CONFIGS = {
    vloMemoryLoadImageBatch: {
        kind: "image",
        inputWidget: "images",
        itemLabel: "image",
    },
    vloMemoryLoadAudioBatch: {
        kind: "audio",
        inputWidget: "audios",
        itemLabel: "audio clip",
    },
    vloMemoryLoadVideoBatch: {
        kind: "video",
        inputWidget: "files",
        itemLabel: "video",
        // Per-item switch delivered next to the ordered media, mirroring the
        // speaker toggles vlo shows on each item of its batch slot.
        flagWidget: "include_audio",
        flagTitle: "Use this video's audio as a reference",
    },
};

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name) ?? null;
}

function markGraphChanged(node) {
    node.graph?.change?.();
    node.graph?.setDirtyCanvas?.(true, true);
    node.setDirtyCanvas?.(true, true);
}

function replaceWidgetAtIndex(node, originalWidget, replacementFactory) {
    const widgets = node.widgets ?? [];
    const originalIndex = widgets.indexOf(originalWidget);
    const originalValue = normalizeBatchSelection(originalWidget.value);

    if (typeof node.removeWidget === "function") {
        node.removeWidget(originalWidget);
    } else if (originalIndex >= 0) {
        originalWidget.onRemove?.();
        widgets.splice(originalIndex, 1);
    }

    const replacement = replacementFactory(originalValue);
    const replacementIndex = node.widgets?.indexOf(replacement) ?? -1;
    // Legacy workflows restore widgets_values by position. Keep the custom
    // widget in the exact slot occupied by the stock MultiSelect; named-value
    // workflows then preserve the same contract as well.
    if (
        originalIndex >= 0 &&
        replacementIndex >= 0 &&
        replacementIndex !== originalIndex
    ) {
        node.widgets.splice(replacementIndex, 1);
        node.widgets.splice(originalIndex, 0, replacement);
    }
    return replacement;
}

function createOrderedBatchSelector(node, config, toggleWidget, flagWidget, initialValue) {
    const root = document.createElement("div");
    root.className = "vlo-batch-selector";
    root.style.setProperty("--comfy-widget-min-height", "210px");
    root.style.setProperty("--comfy-widget-height", "210px");
    root.innerHTML = `
        <style>
            .vlo-batch-selector {
                box-sizing: border-box;
                color: var(--input-text, #ddd);
                font: 12px sans-serif;
                padding: 4px 2px;
                width: 100%;
            }
            .vlo-batch-selector__toolbar,
            .vlo-batch-selector__row {
                align-items: center;
                display: flex;
                gap: 5px;
            }
            .vlo-batch-selector__toolbar select {
                background: var(--comfy-input-bg, #222);
                border: 1px solid var(--border-color, #555);
                border-radius: 4px;
                color: inherit;
                flex: 1;
                min-width: 0;
                padding: 4px;
            }
            .vlo-batch-selector button {
                background: var(--comfy-input-bg, #333);
                border: 1px solid var(--border-color, #555);
                border-radius: 4px;
                color: inherit;
                cursor: pointer;
                padding: 3px 7px;
            }
            .vlo-batch-selector button:disabled {
                cursor: default;
                opacity: 0.45;
            }
            .vlo-batch-selector__summary {
                color: var(--descrip-text, #aaa);
                margin: 5px 1px;
            }
            .vlo-batch-selector__list {
                display: flex;
                flex-direction: column;
                gap: 3px;
                max-height: 140px;
                overflow-y: auto;
            }
            .vlo-batch-selector__row {
                background: color-mix(in srgb, var(--comfy-input-bg, #222) 85%, transparent);
                border-radius: 4px;
                padding: 3px 4px;
            }
            .vlo-batch-selector__index {
                color: var(--descrip-text, #aaa);
                min-width: 20px;
                text-align: right;
            }
            .vlo-batch-selector__name {
                flex: 1;
                min-width: 0;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .vlo-batch-selector__missing {
                color: #e7a85f;
                font-size: 10px;
            }
            .vlo-batch-selector__empty,
            .vlo-batch-selector__status {
                color: var(--descrip-text, #999);
                padding: 5px;
                text-align: center;
            }
        </style>
    `;

    const toolbar = document.createElement("div");
    toolbar.className = "vlo-batch-selector__toolbar";
    const availableSelect = document.createElement("select");
    const addButton = document.createElement("button");
    addButton.type = "button";
    addButton.textContent = "Add";
    const refreshButton = document.createElement("button");
    refreshButton.type = "button";
    refreshButton.title = "Refresh available media";
    refreshButton.textContent = "↻";
    toolbar.append(availableSelect, addButton, refreshButton);

    const summary = document.createElement("div");
    summary.className = "vlo-batch-selector__summary";
    const list = document.createElement("div");
    list.className = "vlo-batch-selector__list";
    const status = document.createElement("div");
    status.className = "vlo-batch-selector__status";
    root.append(toolbar, summary, list, status);

    for (const eventName of ["pointerdown", "mousedown", "wheel"]) {
        root.addEventListener(eventName, (event) => event.stopPropagation());
    }

    let selected = normalizeBatchSelection(initialValue);
    let available = [];
    let loadState = "idle";
    let loadError = null;
    let requestVersion = 0;
    let widget = null;

    const isInputFolderMode = () => Boolean(toggleWidget.value);
    const sourceLabel = () =>
        isInputFolderMode() ? "ComfyUI input folder" : "vlo memory";

    const readFlags = (count) =>
        flagWidget ? normalizeBatchFlags(flagWidget.value, count) : [];

    /**
     * Selection and flags are committed together: a flag belongs to the item
     * at its index, so every add, remove, and move has to carry it along or
     * the loader would deliver someone else's audio.
     */
    const commit = (nextValue, nextFlags) => {
        const normalized = normalizeBatchSelection(nextValue);
        widget.value = normalized;
        selected = normalized;
        if (flagWidget) {
            flagWidget.value = formatBatchFlags(
                normalizeBatchFlags(nextFlags ?? readFlags(normalized.length), normalized.length)
            );
        }
        markGraphChanged(node);
    };

    const render = () => {
        const selectedSet = new Set(selected);
        const availableSet = new Set(available);
        const candidates = available.filter((value) => !selectedSet.has(value));

        availableSelect.replaceChildren();
        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = candidates.length
            ? `Choose ${config.itemLabel}…`
            : `No ${config.itemLabel}s available`;
        availableSelect.append(placeholder);
        for (const value of candidates) {
            const option = document.createElement("option");
            option.value = value;
            option.textContent = value;
            option.title = value;
            availableSelect.append(option);
        }

        const atLimit = selected.length >= MAX_BATCH_SELECTION_ITEMS;
        availableSelect.disabled = loadState === "loading" || !candidates.length || atLimit;
        addButton.disabled = availableSelect.disabled;
        summary.textContent = `${selected.length}/${MAX_BATCH_SELECTION_ITEMS} selected · ${sourceLabel()}`;

        list.replaceChildren();
        if (!selected.length) {
            const empty = document.createElement("div");
            empty.className = "vlo-batch-selector__empty";
            empty.textContent = `No ${config.itemLabel}s selected`;
            list.append(empty);
        }

        selected.forEach((value, index) => {
            const row = document.createElement("div");
            row.className = "vlo-batch-selector__row";

            const ordinal = document.createElement("span");
            ordinal.className = "vlo-batch-selector__index";
            ordinal.textContent = `${index + 1}.`;

            const name = document.createElement("span");
            name.className = "vlo-batch-selector__name";
            name.textContent = value;
            name.title = value;

            row.append(ordinal, name);
            if (loadState === "ready" && !availableSet.has(value)) {
                const unavailable = document.createElement("span");
                unavailable.className = "vlo-batch-selector__missing";
                unavailable.textContent = "unavailable";
                unavailable.title = `Not present in ${sourceLabel()}`;
                row.append(unavailable);
            }

            if (flagWidget) {
                const flags = readFlags(selected.length);
                const flagToggle = document.createElement("input");
                flagToggle.type = "checkbox";
                flagToggle.checked = flags[index] === true;
                flagToggle.title = config.flagTitle ?? "Per-item option";
                flagToggle.addEventListener("change", () => {
                    const nextFlags = readFlags(selected.length);
                    nextFlags[index] = flagToggle.checked;
                    commit(selected, nextFlags);
                });
                row.append(flagToggle);
            }

            const upButton = document.createElement("button");
            upButton.type = "button";
            upButton.textContent = "↑";
            upButton.title = "Move earlier";
            upButton.disabled = index === 0;
            upButton.addEventListener("click", () => {
                commit(
                    moveBatchSelection(selected, index, index - 1),
                    moveBatchSelection(readFlags(selected.length), index, index - 1)
                );
            });

            const downButton = document.createElement("button");
            downButton.type = "button";
            downButton.textContent = "↓";
            downButton.title = "Move later";
            downButton.disabled = index === selected.length - 1;
            downButton.addEventListener("click", () => {
                commit(
                    moveBatchSelection(selected, index, index + 1),
                    moveBatchSelection(readFlags(selected.length), index, index + 1)
                );
            });

            const removeButton = document.createElement("button");
            removeButton.type = "button";
            removeButton.textContent = "×";
            removeButton.title = "Remove";
            removeButton.addEventListener("click", () => {
                const remainingFlags = readFlags(selected.length).filter(
                    (_, itemIndex) => itemIndex !== index
                );
                commit(
                    selected.filter((_, itemIndex) => itemIndex !== index),
                    remainingFlags
                );
            });
            row.append(upButton, downButton, removeButton);
            list.append(row);
        });

        status.hidden = loadState === "ready";
        status.textContent =
            loadState === "loading"
                ? "Loading available media…"
                : loadError
                  ? `Could not load options: ${loadError}`
                  : "";
    };

    const refreshOptions = async () => {
        const version = ++requestVersion;
        loadState = "loading";
        loadError = null;
        render();
        try {
            const response = await api.fetchApi(
                getBatchSourceRoute(config.kind, isInputFolderMode())
            );
            if (!response.ok) {
                throw new Error(`request failed (${response.status})`);
            }
            const payload = await response.json();
            if (version !== requestVersion) return;
            available = normalizeBatchSelection(payload, Number.MAX_SAFE_INTEGER);
            loadState = "ready";
        } catch (error) {
            if (version !== requestVersion) return;
            available = [];
            loadState = "error";
            loadError = error instanceof Error ? error.message : String(error);
        }
        render();
        node.setDirtyCanvas?.(true, false);
    };

    addButton.addEventListener("click", () => {
        if (!availableSelect.value) return;
        commit([...selected, availableSelect.value], [...readFlags(selected.length), false]);
    });
    refreshButton.addEventListener("click", () => void refreshOptions());

    widget = node.addDOMWidget(config.inputWidget, "vlo-media-batch", root, {
        // Returning an Array is intentional. ComfyUI's graphToPrompt wraps
        // array-valued widgets in {__value__: ...}, keeping them distinct from
        // [node_id, output_slot] links before backend validation unwraps them.
        getValue: () => selected.slice(),
        setValue: (value) => {
            selected = normalizeBatchSelection(value);
            render();
        },
        getMinHeight: () => 210,
        getMaxHeight: () => 310,
        getHeight: () => 210,
        hideOnZoom: false,
    });
    widget.value = initialValue;

    const originalToggleCallback = toggleWidget.callback;
    toggleWidget.callback = function (value) {
        originalToggleCallback?.call(this, value);
        void refreshOptions();
    };

    void refreshOptions();
    return widget;
}

app.registerExtension({
    name: "vlo.MemoryLoader.OrderedBatchSelector",
    async nodeCreated(node) {
        const config = BATCH_NODE_CONFIGS[node.comfyClass];
        if (!config) return;

        const inputWidget = findWidget(node, config.inputWidget);
        const toggleWidget = findWidget(node, "disable_in_memory");
        if (!inputWidget || !toggleWidget) {
            console.error(`vlo: could not initialize ${node.comfyClass} batch selector`);
            return;
        }

        // The flag list is written by the per-row checkboxes, so the raw text
        // widget only gets in the way. It stays on the node — hidden widgets
        // still serialize — so the value survives save and reload.
        const flagWidget = config.flagWidget
            ? findWidget(node, config.flagWidget)
            : null;
        if (flagWidget) {
            flagWidget.hidden = true;
            flagWidget.computeSize = () => [0, -4];
        }

        replaceWidgetAtIndex(node, inputWidget, (initialValue) =>
            createOrderedBatchSelector(
                node,
                config,
                toggleWidget,
                flagWidget,
                initialValue
            )
        );
    },
});
