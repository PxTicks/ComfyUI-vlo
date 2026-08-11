import assert from "node:assert/strict";
import test from "node:test";

import {
    getBatchSourceRoute,
    moveBatchSelection,
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
