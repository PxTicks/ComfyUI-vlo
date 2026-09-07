import assert from "node:assert/strict";
import test from "node:test";

import { installRemoteResetGuard } from "../web/memory_widget_guard.mjs";

/** A combo widget shaped like litegraph's: `value` lives on the prototype. */
function makeWidget(initialValue) {
    const proto = {
        get value() {
            return this._state.value;
        },
        set value(next) {
            this._state.value = next;
        },
    };
    const widget = Object.create(proto);
    widget._state = { value: initialValue };
    return widget;
}

function makeGuardedWidget({
    initialValue,
    memoryValues,
    folderMode = false,
}) {
    const widget = makeWidget(initialValue);
    const guard = installRemoteResetGuard({
        widget,
        isFolderMode: () => folderMode,
        readMemoryValues: () => memoryValues,
    });
    return { widget, guard };
}

const NEWEST = "cbe147c8-7779-4463-af43-888a915ebf35";
const RESTORED = "8f0e1d22-4a11-4c9a-9d33-2b7c5e6f0a41";

test("keeps the configured value when the remote list resolves", () => {
    const { widget, guard } = makeGuardedWidget({
        initialValue: RESTORED,
        memoryValues: [NEWEST, RESTORED],
    });
    guard.noteRestoredValue();

    // What useRemoteWidget's onFirstLoad does.
    widget.value = NEWEST;

    assert.equal(widget.value, RESTORED);
});

test("two loaders sharing one options route keep their own selections", () => {
    const source = makeGuardedWidget({
        initialValue: "source-media-id",
        memoryValues: [NEWEST, "source-media-id", "mask-media-id"],
    });
    const mask = makeGuardedWidget({
        initialValue: "mask-media-id",
        memoryValues: [NEWEST, "source-media-id", "mask-media-id"],
    });
    source.guard.noteRestoredValue();
    mask.guard.noteRestoredValue();

    source.widget.value = NEWEST;
    mask.widget.value = NEWEST;

    assert.equal(source.widget.value, "source-media-id");
    assert.equal(mask.widget.value, "mask-media-id");
});

test("absorbs the reset only once, so later picks stick", () => {
    const { widget, guard } = makeGuardedWidget({
        initialValue: RESTORED,
        memoryValues: [NEWEST, RESTORED],
    });
    guard.noteRestoredValue();

    widget.value = NEWEST;
    assert.equal(widget.value, RESTORED);

    // A deliberate pick of the same entry afterwards is honoured.
    widget.value = NEWEST;
    assert.equal(widget.value, NEWEST);
});

test("passes through writes that are not the list's first entry", () => {
    const { widget, guard } = makeGuardedWidget({
        initialValue: RESTORED,
        memoryValues: [NEWEST, "picked-media-id"],
    });
    guard.noteRestoredValue();

    widget.value = "picked-media-id";

    assert.equal(widget.value, "picked-media-id");
});

test("leaves a freshly dropped node free to take the remote default", () => {
    const { widget } = makeGuardedWidget({
        initialValue: "Loading...",
        memoryValues: [NEWEST],
    });
    // No noteRestoredValue(): the node was never configured from a workflow.

    widget.value = NEWEST;

    assert.equal(widget.value, NEWEST);
});

test("stays out of the way before the option list resolves", () => {
    // useRemoteWidget hands back a bare default, not an array, until then.
    const { widget, guard } = makeGuardedWidget({
        initialValue: RESTORED,
        memoryValues: "Loading...",
    });
    guard.noteRestoredValue();

    widget.value = NEWEST;

    assert.equal(widget.value, NEWEST);
});

test("defers to the folder-mode reconcile pass", () => {
    const { widget, guard } = makeGuardedWidget({
        initialValue: RESTORED,
        memoryValues: [NEWEST, RESTORED],
        folderMode: true,
    });
    guard.noteRestoredValue();

    widget.value = NEWEST;

    assert.equal(widget.value, NEWEST);
});

test("wraps a plain data property when the widget has no accessor", () => {
    const widget = { value: RESTORED };
    const guard = installRemoteResetGuard({
        widget,
        isFolderMode: () => false,
        readMemoryValues: () => [NEWEST, RESTORED],
    });
    guard.noteRestoredValue();

    widget.value = NEWEST;

    assert.equal(widget.value, RESTORED);
});
