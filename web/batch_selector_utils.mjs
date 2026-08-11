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
