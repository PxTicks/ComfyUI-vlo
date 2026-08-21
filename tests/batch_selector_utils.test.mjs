import assert from "node:assert/strict";
import test from "node:test";

import {
    formatBatchFlags,
    getBatchSourceRoute,
    moveBatchSelection,
    normalizeBatchFlags,
    normalizeBatchSelection,
} from "../web/batch_selector_utils.mjs";

test("normalizes, deduplicates, and caps UI selections in order", () => {
    assert.deepEqual(normalizeBatchSelection([" b ", "a", "b", 4], 2), [
        "b",
        "a",
    ]);
});

test("moves one selection without mutating the source", () => {
    const source = ["a", "b", "c"];
    assert.deepEqual(moveBatchSelection(source, 2, 0), ["c", "a", "b"]);
    assert.deepEqual(source, ["a", "b", "c"]);
});

test("builds the registry-first and input-folder routes", () => {
    assert.equal(
        getBatchSourceRoute("image", false),
        "/api/vlo-memory/options?kind=image"
    );
    assert.equal(
        getBatchSourceRoute("audio clip", true),
        "/api/vlo-memory/input-files?kind=audio%20clip"
    );
});

test("pads and trims per-item flags to the selection length", () => {
    assert.deepEqual(normalizeBatchFlags("1,0", 3), [true, false, false]);
    assert.deepEqual(normalizeBatchFlags("1,1,1", 2), [true, true]);
    assert.deepEqual(normalizeBatchFlags("", 2), [false, false]);
    assert.deepEqual(normalizeBatchFlags([true, "0", 1], 3), [
        true,
        false,
        true,
    ]);
});

test("moves flags alongside the selection they belong to", () => {
    const selection = ["a", "b", "c"];
    const flags = normalizeBatchFlags("0,1,0", selection.length);
    assert.equal(
        formatBatchFlags(moveBatchSelection(flags, 1, 0)),
        "1,0,0"
    );
    assert.deepEqual(moveBatchSelection(selection, 1, 0), ["b", "a", "c"]);
});
