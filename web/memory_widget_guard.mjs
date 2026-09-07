/** Protects a memory loader's file widget from ComfyUI's remote-combo reset.
 *
 * ComfyUI's remote-combo helper (useRemoteWidget's `onFirstLoad`) writes the
 * first fetched option into `widget.value` the first time an option list
 * resolves, discarding whatever the workflow restored. Every vlo memory loader
 * of a given kind shares one options route, and that route lists the registry
 * newest-first, so a graph holding a source loader and a mask loader ends up
 * with both silently repointed at the same, newest clip.
 */

/** Resolve `value`'s accessor wherever it lives — BaseWidget defines it on the
 *  prototype and may delegate to a widget-value store, so installing a plain
 *  own property would shadow that machinery rather than wrap it. */
export function findValueDescriptor(widget) {
    let target = widget;
    while (target) {
        const descriptor = Object.getOwnPropertyDescriptor(target, "value");
        if (descriptor) return descriptor;
        target = Object.getPrototypeOf(target);
    }
    return null;
}

/**
 * Wrap `widget.value` so the remote helper's one-shot reset is absorbed.
 *
 * @param {object} options
 * @param {object} options.widget            the file/image/audio combo widget
 * @param {() => boolean} options.isFolderMode  true while `disable_in_memory`
 *   is on; folder mode has its own reconcile pass, and reading the memory list
 *   here would kick off a fetch it does not want
 * @param {() => unknown} options.readMemoryValues  the registry-backed option
 *   list, as the remote helper would see it
 * @returns {{ noteRestoredValue: () => void } | null} null when the widget has
 *   no `value` accessor to wrap
 */
export function installRemoteResetGuard({
    widget,
    isFolderMode,
    readMemoryValues,
}) {
    const descriptor = findValueDescriptor(widget);
    if (!descriptor) return null;

    let localValue = descriptor.get ? undefined : descriptor.value;
    const readValue = descriptor.get
        ? () => descriptor.get.call(widget)
        : () => localValue;
    const writeValue = descriptor.set
        ? (value) => descriptor.set.call(widget, value)
        : (value) => {
              localValue = value;
          };

    // The value the serialized workflow asked for. Null until `onConfigure`
    // reports one, which leaves a freshly dropped node free to take whatever
    // default the remote list offers.
    let restoredValue = null;
    let absorbedRemoteReset = false;

    const isRemoteFirstLoadReset = (next) => {
        if (absorbedRemoteReset) return false;
        if (isFolderMode()) return false;
        if (typeof restoredValue !== "string" || !restoredValue) return false;
        if (next === restoredValue) return false;
        // `onFirstLoad` writes exactly the list's first entry. Matching on that
        // keeps the guard from swallowing a user's pick or an upload that lands
        // while the fetch is still in flight. Before the list resolves the
        // helper hands back a bare default rather than an array, so the guard
        // cannot arm early either.
        const memoryValues = readMemoryValues();
        return (
            Array.isArray(memoryValues) &&
            memoryValues.length > 0 &&
            next === memoryValues[0]
        );
    };

    Object.defineProperty(widget, "value", {
        configurable: true,
        enumerable: true,
        get() {
            return readValue();
        },
        set(next) {
            if (isRemoteFirstLoadReset(next)) {
                absorbedRemoteReset = true;
                return;
            }
            writeValue(next);
        },
    });

    return {
        /** Call once the node has been configured from the workflow. */
        noteRestoredValue() {
            const configured = readValue();
            if (typeof configured === "string" && configured) {
                restoredValue = configured;
            }
        },
    };
}
