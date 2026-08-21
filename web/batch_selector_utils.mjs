export const MAX_BATCH_SELECTION_ITEMS = 100;

export function normalizeBatchSelection(
    value,
    maxItems = MAX_BATCH_SELECTION_ITEMS
) {
    const candidates = Array.isArray(value)
        ? value
        : typeof value === "string" && value.trim()
          ? [value]
          : [];
    const normalized = [];
    const seen = new Set();

    for (const candidate of candidates) {
        if (typeof candidate !== "string") continue;
        const item = candidate.trim();
        if (!item || seen.has(item)) continue;
        seen.add(item);
        normalized.push(item);
        if (normalized.length >= maxItems) break;
    }

    return normalized;
}

export function moveBatchSelection(values, fromIndex, toIndex) {
    if (
        fromIndex < 0 ||
        fromIndex >= values.length ||
        toIndex < 0 ||
        toIndex >= values.length ||
        fromIndex === toIndex
    ) {
        return values.slice();
    }

    const next = values.slice();
    const [item] = next.splice(fromIndex, 1);
    next.splice(toIndex, 0, item);
    return next;
}

export function getBatchSourceRoute(kind, disableInMemory) {
    const encodedKind = encodeURIComponent(kind);
    return disableInMemory
        ? `/api/vlo-memory/input-files?kind=${encodedKind}`
        : `/api/vlo-memory/options?kind=${encodedKind}`;
}

const TRUE_FLAG_TOKENS = new Set(["1", "true", "yes", "on"]);

/**
 * Reads the loader's comma-separated flag widget into exactly `count`
 * booleans. Anything the user never set reads as false, which is what the
 * Python side assumes too.
 */
export function normalizeBatchFlags(value, count) {
    const tokens = Array.isArray(value)
        ? value
        : typeof value === "string" && value.trim()
          ? value.split(",")
          : [];
    const flags = [];
    for (let index = 0; index < count; index += 1) {
        const token = tokens[index];
        if (typeof token === "boolean") {
            flags.push(token);
            continue;
        }
        if (typeof token === "number") {
            flags.push(token !== 0);
            continue;
        }
        flags.push(
            typeof token === "string" &&
                TRUE_FLAG_TOKENS.has(token.trim().toLowerCase())
        );
    }
    return flags;
}

export function formatBatchFlags(flags) {
    return flags.map((flag) => (flag ? "1" : "0")).join(",");
}
