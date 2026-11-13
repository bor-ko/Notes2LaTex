import katex from "katex";
import { useMemo } from "react";

// Approximations of the LaTeX preamble macros, for a quick in-browser preview.
// The compiled PDF is the real rendering.
const MACROS: Record<string, string> = {
  // No "#rrggbb" here: inside a macro KaTeX reads # as a parameter marker.
  "\\unsure": "\\textcolor{darkorange}{#1}",
  "\\dd": "\\mathrm{d}",
  "\\vec": "\\boldsymbol{#1}",
  "\\qty": "#1\\,\\mathrm{#2}",
  "\\tr": "\\operatorname{tr}",
};

function math(src: string, display: boolean): string {
  return katex.renderToString(src, { displayMode: display, throwOnError: false, strict: false, macros: { ...MACROS } });
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function textPart(s: string): string {
  let out = escapeHtml(s);
  const wrap = (cmd: string, open: string, close: string) => {
    out = out.replace(new RegExp(`\\\\${cmd}\\{([^{}]*)\\}`, "g"), `${open}$1${close}`);
  };
  wrap("unsure", '<mark class="unsure">', "</mark>");
  wrap("textbf", "<b>", "</b>");
  wrap("emph", "<i>", "</i>");
  wrap("textit", "<i>", "</i>");
  wrap("texttt", "<code>", "</code>");
  wrap("underline", "<u>", "</u>");
  return out.replace(/\\item\s*/g, "• ").replace(/\\\\/g, "<br/>").replace(/\\([&%$#_{}])/g, "$1");
}

/** Text with $...$ / \(...\) inline math. */
function mixed(src: string): string {
  const parts = src.split(/(\$\$[\s\S]+?\$\$|\$[^$]+\$|\\\([\s\S]+?\\\)|\\\[[\s\S]+?\\\])/g);
  return parts
    .map((p) => {
      if (p.startsWith("$$")) return math(p.slice(2, -2), true);
      if (p.startsWith("$")) return math(p.slice(1, -1), false);
      if (p.startsWith("\\(") || p.startsWith("\\[")) return math(p.slice(2, -2), p.startsWith("\\["));
      return textPart(p);
    })
    .join("");
}

export function LatexPreview({ role, latex }: { role: string; latex: string }) {
  const html = useMemo(() => {
    const src = latex ?? "";
    if (!src.trim()) return role === "diagram" ? '<span class="muted">Figure: the crop is inserted as an image.</span>' : '<span class="muted">(empty)</span>';
    switch (role) {
      case "equation":
        return math(src, true);
      case "derivation":
        return math(`\\begin{aligned}${src}\\end{aligned}`, true);
      case "code":
      case "pseudocode":
        return `<pre>${escapeHtml(src)}</pre>`;
      case "heading":
        return `<h3>${mixed(src)}</h3>`;
      case "subheading":
        return `<h4>${mixed(src)}</h4>`;
      default:
        return `<div>${mixed(src)}</div>`;
    }
  }, [role, latex]);
  return <div className={`preview role-${role}`} dangerouslySetInnerHTML={{ __html: html }} />;
}
