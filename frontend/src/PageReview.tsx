import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { StreamLanguage } from "@codemirror/language";
import { stex } from "@codemirror/legacy-modes/mode/stex";
import { EditorView } from "@codemirror/view";
import { api, Block, BlockEdit, PageDetail, Status, toEdit } from "./api";
import { go } from "./App";
import { LatexPreview } from "./LatexPreview";

const ROLE_COLORS: Record<string, string> = {
  heading: "#7c3aed",
  subheading: "#9f67fa",
  definition: "#2563eb",
  theorem: "#dc2626",
  proof: "#64748b",
  example: "#16a34a",
  note: "#ca8a04",
  equation: "#0d9488",
  derivation: "#0891b2",
  prose: "#475569",
  list: "#475569",
  code: "#4f46e5",
  pseudocode: "#6366f1",
  diagram: "#db2777",
};
const UNSURE = "#d97706";
const latexLang = StreamLanguage.define(stex);
const editorExt = [latexLang, EditorView.lineWrapping];

function usePrefersDark(): boolean {
  const query = "(prefers-color-scheme: dark)";
  const [dark, setDark] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const on = () => setDark(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return dark;
}

export function PageReview({ id, status, engine, focusBlock }: { id: number; status: Status; engine: string; focusBlock: number | null }) {
  const [page, setPage] = useState<PageDetail | null>(null);
  const [edits, setEdits] = useState<BlockEdit[]>([]);
  const [dirty, setDirty] = useState(false);
  const [selected, setSelected] = useState<number | null>(focusBlock);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [splitFor, setSplitFor] = useState<number | null>(null);
  const cardRefs = useRef(new Map<number, HTMLDivElement>());

  const apply = useCallback((p: PageDetail) => {
    setPage(p);
    setEdits(p.blocks.map(toEdit));
    setDirty(false);
  }, []);

  useEffect(() => {
    api.page(id).then(apply, (e) => setErr(e.message));
  }, [id, apply]);

  // Poll while this page is being transcribed.
  useEffect(() => {
    if (page?.status !== "transcribing") return;
    const t = setInterval(() => api.page(id).then(apply), 1500);
    return () => clearInterval(t);
  }, [page?.status, id, apply]);

  useEffect(() => {
    if (focusBlock) scrollTo(focusBlock);
  }, [page?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const blocksById = useMemo(() => new Map(page?.blocks.map((b) => [b.id, b]) ?? []), [page]);

  function scrollTo(blockId: number) {
    setSelected(blockId);
    cardRefs.current.get(blockId)?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function update(blockId: number, patch: Partial<BlockEdit>) {
    setEdits((es) => es.map((e) => (e.id === blockId ? { ...e, ...patch } : e)));
    setDirty(true);
  }

  async function run(label: string, fn: () => Promise<PageDetail | void>) {
    setBusy(label);
    setErr(null);
    try {
      const p = await fn();
      if (p) apply(p);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  const save = () => run("Saving", () => api.saveBlocks(id, edits));
  const saveIfDirty = async () => {
    if (dirty) apply(await api.saveBlocks(id, edits));
  };
  const approve = () =>
    run("Approving", async () => {
      const p = await api.approve(id, edits);
      if (p.next_id) {
        go(`/page/${p.next_id}`);
        return;
      }
      return p;
    });
  const navigate = (pageId: number | null) =>
    pageId && run("Saving", async () => {
      await saveIfDirty();
      go(`/page/${pageId}`);
    });
  const retranscribe = () => {
    if (!confirm("Transcribe this page again? Your edits on this page will be replaced.")) return;
    run("Queuing", () => api.retranscribe(id, engine));
  };
  const mergeNext = (blockId: number) =>
    run("Merging", async () => {
      await saveIfDirty();
      return api.mergeNext(blockId);
    });
  const split = (blockId: number, at: number) =>
    run("Splitting", async () => {
      setSplitFor(null);
      await saveIfDirty();
      return api.split(blockId, at);
    });

  function nextUnsure() {
    const order = edits.map((e) => e.id);
    const start = selected ? order.indexOf(selected) + 1 : 0;
    const rotated = [...order.slice(start), ...order.slice(0, start)];
    const hit = rotated.find((bid) => edits.find((e) => e.id === bid)?.latex.includes("\\unsure{"));
    if (hit) scrollTo(hit);
  }

  // Keyboard: Cmd/Ctrl+S save, Cmd/Ctrl+Enter approve, Cmd/Ctrl+J next unsure.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey)) return;
      if (e.key === "s") {
        e.preventDefault();
        save();
      } else if (e.key === "Enter") {
        e.preventDefault();
        approve();
      } else if (e.key === "j") {
        e.preventDefault();
        nextUnsure();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // Warn before closing the tab with unsaved edits.
  useEffect(() => {
    if (!dirty) return;
    const on = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", on);
    return () => window.removeEventListener("beforeunload", on);
  }, [dirty]);

  if (err && !page) return <p className="banner error" style={{ margin: 20 }}>{err}</p>;
  if (!page) return <p className="muted" style={{ padding: 20 }}>Loading…</p>;

  const unsureCount = edits.filter((e) => e.latex.includes("\\unsure{")).length;
  const transcribing = page.status === "transcribing";

  return (
    <>
      <div className="review-bar">
        <a href={`#/doc/${page.document.id}`}>{page.document.name}</a>
        <span className="muted">/</span>
        <b>
          Page {page.page_index + 1} of {page.page_count}
        </b>
        <span className={`chip ${page.status}`}>{page.status}</span>
        {page.model && <span className="small muted">{page.model}</span>}
        {unsureCount > 0 && (
          <button className="small" onClick={nextUnsure} title="Cmd/Ctrl+J">
            {unsureCount} unsure → next
          </button>
        )}
        <span style={{ flex: 1 }} />
        {busy && <span className="small muted">{busy}…</span>}
        {dirty && !busy && <span className="small edited">unsaved changes</span>}
        <button onClick={() => navigate(page.prev_id)} disabled={!page.prev_id}>
          ← Prev
        </button>
        <button onClick={() => navigate(page.next_id)} disabled={!page.next_id}>
          Next →
        </button>
        <button onClick={retranscribe} disabled={transcribing}>
          Re-transcribe
        </button>
        <button onClick={save} disabled={!dirty || !!busy} title="Cmd/Ctrl+S">
          Save
        </button>
        <button className="primary" onClick={approve} disabled={!!busy || transcribing || page.status === "new"} title="Cmd/Ctrl+Enter">
          {page.status === "reviewed" ? "Approve again" : "Approve page"}
          {page.next_id ? " & next" : ""}
        </button>
      </div>
      {(err || page.error) && (
        <div className="banner error" style={{ margin: "10px 20px 0" }}>
          {err ?? page.error}
        </div>
      )}
      <div className="review">
        <div className="page-image">
          <div className="frame">
            <img src={page.image_url} alt="Handwritten page" />
            <svg viewBox={`0 0 ${page.width} ${page.height}`} preserveAspectRatio="none">
              {page.blocks.map((b, i) => {
                const e = edits.find((x) => x.id === b.id);
                const unsure = e?.latex.includes("\\unsure{");
                const color = unsure ? UNSURE : ROLE_COLORS[e?.role ?? b.role] ?? "#475569";
                return (
                  <g key={b.id} onClick={() => scrollTo(b.id)}>
                    <rect
                      className={selected === b.id ? "selected" : ""}
                      x={b.x0}
                      y={b.y0}
                      width={b.x1 - b.x0}
                      height={b.y1 - b.y0}
                      stroke={color}
                      fill={color}
                      rx={6}
                    />
                    <text x={Math.max(4, b.x0 - 34)} y={b.y0 + 24} fill={color}>
                      {i + 1}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        </div>
        <div className="blocks">
          {transcribing && <div className="banner">Transcribing this page… it updates automatically.</div>}
          {page.status === "new" && <div className="banner">This page has not been transcribed yet. Use "Re-transcribe".</div>}
          {page.blocks.length === 0 && <p className="muted">No writing was found on this page.</p>}
          {page.blocks.map((b, i) => {
            const e = edits.find((x) => x.id === b.id);
            if (!e) return null;
            return (
              <BlockCard
                key={b.id}
                index={i + 1}
                block={b}
                edit={e}
                roles={status.roles}
                selected={selected === b.id}
                isLast={i === page.blocks.length - 1}
                splitMode={splitFor === b.id}
                refCb={(el) => (el ? cardRefs.current.set(b.id, el) : cardRefs.current.delete(b.id))}
                onSelect={() => setSelected(b.id)}
                onChange={(patch) => update(b.id, patch)}
                onMerge={() => mergeNext(b.id)}
                onToggleSplit={() => setSplitFor(splitFor === b.id ? null : b.id)}
                onSplit={(at) => split(b.id, at)}
                disabled={!!busy || transcribing}
                draft={blocksById.get(b.id)?.draft_latex ?? null}
              />
            );
          })}
        </div>
      </div>
    </>
  );
}

type CardProps = {
  index: number;
  block: Block;
  edit: BlockEdit;
  roles: string[];
  selected: boolean;
  isLast: boolean;
  splitMode: boolean;
  disabled: boolean;
  draft: string | null;
  refCb: (el: HTMLDivElement | null) => void;
  onSelect: () => void;
  onChange: (patch: Partial<BlockEdit>) => void;
  onMerge: () => void;
  onToggleSplit: () => void;
  onSplit: (at: number) => void;
};

function BlockCard(p: CardProps) {
  const { block, edit } = p;
  const dark = usePrefersDark();
  const unsure = edit.latex.includes("\\unsure{");
  const edited = p.draft !== null && (edit.latex !== p.draft || edit.role !== block.draft_role);
  const classes = ["block", p.selected && "selected", unsure && "unsure", edit.continues_previous && "continues"].filter(Boolean).join(" ");

  return (
    <div className={classes} ref={p.refCb} onFocusCapture={p.onSelect} onClick={p.onSelect}>
      <div className="block-head">
        <span className="num">{p.index}</span>
        <select value={edit.role} onChange={(e) => p.onChange({ role: e.target.value })} disabled={p.disabled}>
          {p.roles.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
        <label className="check small" title="This line continues the element above (same paragraph, derivation, listing…)">
          <input
            type="checkbox"
            checked={edit.continues_previous}
            disabled={p.disabled}
            onChange={(e) => p.onChange({ continues_previous: e.target.checked })}
          />
          continues above
        </label>
        {edit.role === "code" && (
          <input
            type="text"
            className="small"
            style={{ width: 110 }}
            placeholder="language"
            value={edit.code_language ?? ""}
            disabled={p.disabled}
            onChange={(e) => p.onChange({ code_language: e.target.value || null })}
          />
        )}
        <span className="grow" />
        {edited && <span className="edited">edited</span>}
        <button className="small" onClick={p.onToggleSplit} disabled={p.disabled} title="Segmentation joined two lines? Split this block.">
          {p.splitMode ? "Cancel split" : "Split"}
        </button>
        <button className="small" onClick={p.onMerge} disabled={p.disabled || p.isLast} title="Segmentation cut one line in two? Merge with the next block.">
          Merge ↓
        </button>
      </div>
      <div
        className={`crop${p.splitMode ? " split-mode" : ""}`}
        onClick={(e) => {
          if (!p.splitMode) return;
          const r = (e.target as HTMLElement).getBoundingClientRect();
          p.onSplit((e.clientY - r.top) / r.height);
        }}
        title={p.splitMode ? "Click where the split should be" : undefined}
      >
        <img src={block.crop_url} alt={`Line ${p.index}`} />
      </div>
      <CodeMirror
        value={edit.latex}
        theme={dark ? "dark" : "light"}
        extensions={editorExt}
        editable={!p.disabled}
        basicSetup={{ lineNumbers: false, foldGutter: false, highlightActiveLine: false }}
        onChange={(v) => p.onChange({ latex: v })}
      />
      <LatexPreview role={edit.role} latex={edit.latex} />
    </div>
  );
}
