/* ─── Formatting helpers shared by components ────────────── */

/** The API returns the overlay as bare base64 PNG; history stores a JPEG data URL. */
export function overlaySrc(b64: string | null): string | null {
  if (!b64) return null;
  return b64.startsWith('data:') ? b64 : `data:image/png;base64,${b64}`;
}

/** First top-level heading, stripped of markdown — used as a card title. */
export function markdownTitle(text: string): string | null {
  const match = /^#\s+(.+)$/m.exec(text);
  return match ? stripMarkdown(match[1]) : null;
}

/** Plain-text preview: no headings, no emphasis markers, collapsed whitespace. */
export function markdownPreview(text: string, max = 220): string {
  const plain = stripMarkdown(
    text
      .split('\n')
      .filter(l => !/^\s*#/.test(l))
      .join(' '),
  );
  return plain.length > max ? `${plain.slice(0, max).replace(/\s+\S*$/, '')}…` : plain;
}

function stripMarkdown(s: string): string {
  return s
    .replace(/^\s*>\s?/gm, '')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/[*_]([^*_]+)[*_]/g, '$1')
    .replace(/^\s*([-*•]|\d+[.)])\s+/gm, '')
    .replace(/\s+/g, ' ')
    .trim();
}
