import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CONFIGS = {
    vloMemoryLoadImage: { kind: "image", fileWidget: "image" },
    vloMemoryLoadAudio: { kind: "audio", fileWidget: "audio" },
    vloMemoryLoadVideo: { kind: "video", fileWidget: "file" },
};

function findWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

app.registerExtension({
    name: "vlo.MemoryLoader.InputFolderDropdown",
    async nodeCreated(node) {
        const config = NODE_CONFIGS[node.comfyClass];
        if (!config) return;

        const fileWidget = findWidget(node, config.fileWidget);
        const toggleWidget = findWidget(node, "disable_in_memory");
        if (!fileWidget || !toggleWidget) return;

        let inputFiles = null;
        let inputFetchPromise = null;

        const fetchInputFiles = () => {
            if (inputFetchPromise) return inputFetchPromise;
            inputFetchPromise = (async () => {
                try {
                    const resp = await api.fetchApi(
                        `/api/vlo-memory/input-files?kind=${encodeURIComponent(config.kind)}`
                    );
                    if (!resp.ok) {
                        console.error(
                            `vlo: input-files request failed (${resp.status})`
                        );
                        return;
                    }
                    const data = await resp.json();
                    inputFiles = Array.isArray(data) ? data : [];
                    reconcileFileWidgetValue();
                    node.setDirtyCanvas?.(true, false);
                } catch (err) {
                    console.error("vlo: failed to fetch input folder files", err);
                } finally {
                    inputFetchPromise = null;
                }
            })();
            return inputFetchPromise;
        };

        // Wrap the existing `values` descriptor (installed by the remote-widget
        // setup for the memory source) so we can swap sources based on the toggle.
        // If the original descriptor has no setter, fall back to a local cache so
        // writes from ComfyUI's remote refresh aren't silently dropped.
        const options = fileWidget.options;
        const descriptor = Object.getOwnPropertyDescriptor(options, "values");
        let localMemoryValues = Array.isArray(descriptor?.value)
            ? descriptor.value.slice()
            : null;
        const readMemoryValues = descriptor?.get
            ? () => descriptor.get.call(options)
            : () => (localMemoryValues !== null ? localMemoryValues : []);
        const writeMemoryValues = descriptor?.set
            ? (value) => descriptor.set.call(options, value)
            : (value) => {
                  localMemoryValues = Array.isArray(value) ? value.slice() : null;
              };

        Object.defineProperty(options, "values", {
            configurable: true,
            enumerable: true,
            get() {
                if (toggleWidget.value) {
                    if (inputFiles === null) {
                        void fetchInputFiles();
                        return [];
                    }
                    return inputFiles;
                }
                return readMemoryValues();
            },
            set(value) {
                writeMemoryValues(value);
            },
        });

        // When input-folder mode is active, ensure the file widget's `value`
        // stays a string from the folder-backed options list. Without this, the
        // serialized widget value can drift into non-string territory and
        // ComfyUI can embed broken workflow metadata for these nodes.
        const reconcileFileWidgetValue = () => {
            // Skip while the input-folder list is still loading; the post-fetch
            // path will call us again with real options.
            if (toggleWidget.value && inputFiles === null) return;

            const validOptions = options.values;
            if (!Array.isArray(validOptions)) return;

            const currentValue = fileWidget.value;
            const isStringValue =
                typeof currentValue === "string" && currentValue.length > 0;
            if (isStringValue && validOptions.includes(currentValue)) return;

            const replacement =
                validOptions.length > 0 ? validOptions[0] : "";
            if (fileWidget.value === replacement) return;
            fileWidget.value = replacement;
            node.setDirtyCanvas?.(true, true);
        };

        const originalToggleCallback = toggleWidget.callback;
        toggleWidget.callback = function (value) {
            originalToggleCallback?.call(this, value);
            if (value) {
                inputFiles = null;
                void fetchInputFiles();
            }
            node.setDirtyCanvas?.(true, true);
        };

        const originalRefresh = fileWidget.refresh;
        fileWidget.refresh = function () {
            originalRefresh?.call(this);
            if (toggleWidget.value) {
                inputFiles = null;
                void fetchInputFiles();
            }
        };

        if (toggleWidget.value) {
            void fetchInputFiles();
        }
    },
});
