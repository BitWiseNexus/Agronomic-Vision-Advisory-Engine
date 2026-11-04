/* ─── Markdown — just enough for retrieved extension documents ─
   Evidence chunks arrive as raw markdown (headings, bullets, bold,
   italics). This renders that subset to React elements — no HTML
   injection, so retrieved text can never execute anything.
   ──────────────────────────────────────────────────────────── */

import type { ReactNode } from 'react';

function inline(text: string, keyBase: string): ReactNode[] {
  // **bold** first, then *italic* / _italic_ inside the remaining runs
  const out: ReactNode[] = [];
  text.split(/(\*\*[^*]+\*\*)/g).forEach((part, i) => {
    if (/^\*\*[^*]+\*\*$/.test(part)) {
      out.push(<strong key={`${keyBase}-b${i}`}>{part.slice(2, -2)}</strong>);
      return;
    }
    part.split(/(\*[^*\s][^*]*\*|_[^_\s][^_]*_)/g).forEach((sub, j) => {
      if (/^(\*[^*]+\*|_[^_]+_)$/.test(sub)) out.push(<em key={`${keyBase}-i${i}-${j}`}>{sub.slice(1, -1)}</em>);
      else if (sub) out.push(sub);
    });
  });
  return out;
}

export function Markdown({ text, skipTitle = false }: { text: string; skipTitle?: boolean }) {
  const blocks: ReactNode[] = [];
  let list: string[] = [];
  let para: string[] = [];
  let titleSkipped = !skipTitle;

  const flushPara = () => {
    if (para.length) blocks.push(<p key={`p${blocks.length}`}>{inline(para.join(' '), `p${blocks.length}`)}</p>);
    para = [];
  };
  const flushList = () => {
    if (list.length) {
      const k = `l${blocks.length}`;
      blocks.push(<ul key={k}>{list.map((item, i) => <li key={i}>{inline(item, `${k}-${i}`)}</li>)}</ul>);
    }
    list = [];
  };

  for (const raw of text.split('\n')) {
    const line = raw.trim().replace(/^>\s?/, ''); // blockquotes read as plain text
    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    const bullet = /^([-*•]|\d+[.)])\s+(.*)$/.exec(line);
    if (!line) {
      flushPara();
      flushList();
    } else if (heading) {
      flushPara();
      flushList();
      if (!titleSkipped && heading[1].length === 1) {
        titleSkipped = true;
        continue;
      }
      blocks.push(<h5 key={`h${blocks.length}`}>{inline(heading[2], `h${blocks.length}`)}</h5>);
    } else if (bullet) {
      flushPara();
      list.push(bullet[2]);
    } else {
      flushList();
      para.push(line);
    }
  }
  flushPara();
  flushList();
  return <div className="md">{blocks}</div>;
}
