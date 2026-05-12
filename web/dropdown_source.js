import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CONFIGS = {
    VLOMemoryLoadImage: { kind: "image", fileWidget: "image" },
    VLOMemoryLoadAudio: { kind: "audio", fileWidget: "audio" },
    VLOMemoryLoadVideo: { kind: "video", fileWidget: "file" },
};

function findWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

app.registerExtension({
    name: "VLO.MemoryLoader.InputFolderDropdown",
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
                            `VLO: input-files request failed (${resp.status})`
                        );
                        return;
                    }
                    const data = await resp.json();
                    inputFiles = Array.isArray(data) ? data : [];
                    node.setDirtyCanvas?.(true, false);
                } catch (err) {
                    console.error("VLO: failed to fetch input folder files", err);
                } finally {
                    inputFetchPromise = null;
                }
            })();
            return inputFetchPromise;
        };

        // Wrap the existing `values` descriptor (installed by the remote-widget
        // setup for the memory source) so we can swap sources based on the toggle.
        const options = fileWidget.options;
        const descriptor = Object.getOwnPropertyDescriptor(options, "values");
        const readMemoryValues = descriptor?.get
            ? () => descriptor.get.call(options)
            : () => (Array.isArray(descriptor?.value) ? descriptor.value : []);
        const writeMemoryValues = descriptor?.set
            ? (value) => descriptor.set.call(options, value)
            : null;

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
                writeMemoryValues?.(value);
            },
        });

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
