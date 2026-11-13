import { useCallback, useEffect, useState } from "react";
import { api, BuildResult, DocDetail, Status } from "./api";
import { go } from "./App";

export function DocumentView({ id, status, engine }: { id: number; status: Status; engine: string }) {
  const [doc, setDoc] = useState<DocDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [building, setBuilding] = useState(false);
  const [build, setBuild] = useState<BuildResult | null>(null);

  const load = useCallback(() => api.document(id).then(setDoc, (e) => setErr(e.message)), [id]);
  useEffect(() => {
    load();
  }, [load]);

  // Poll while pages are being transcribed.
  const working = doc?.pages.some((p) => p.status === "transcribing");
  useEffect(() => {
    if (!working) return;
    const t = setInterval(load, 1500);
    return () => clearInterval(t);
  }, [working, load]);

  if (err) return <p className="banner error">{err}</p>;
  if (!doc) return <p className="muted">Loading…</p>;

  const pending = doc.pages.filter((p) => p.status === "new" || p.status === "error").length;
  const reviewed = doc.pages.filter((p) => p.status === "reviewed").length;
  const cost = doc.pages.reduce((s, p) => s + (p.cost ?? 0), 0);
  const firstToReview = doc.pages.find((p) => p.status === "transcribed") ?? doc.pages[0];
  const pdfUrl = build?.pdf_url ?? doc.pdf_url;

  async function runBuild(final: boolean) {
    setBuilding(true);
    setErr(null);
    try {
      setBuild(await api.build(id, final, engine));
      load();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBuilding(false);
    }
  }

  async function transcribe() {
    try {
      await api.transcribeDocument(id, engine);
      load();
    } catch (e) {
      setErr((e as Error).message);
    }
  }

  async function remove() {
    if (!confirm(`Delete "${doc!.name}" and all its pages and corrections? This cannot be undone.`)) return;
    await api.deleteDocument(id);
    go("/");
  }

  return (
    <div className="stack">
      <div className="row">
        <a href="#/">Library</a>
        <span className="muted">/</span>
        <h1>{doc.name}</h1>
        <span className="chip">{status.subjects.find((s) => s.key === doc.subject)?.name ?? doc.subject}</span>
        <span className="chip split" title="Test documents are held out to measure accuracy honestly.">
          {doc.split}
        </span>
      </div>

      <div className="card row">
        <span>
          {reviewed}/{doc.pages.length} pages reviewed
          {cost > 0 && <span className="muted"> · API cost so far ~${cost.toFixed(2)}</span>}
        </span>
        <span style={{ flex: 1 }} />
        {pending > 0 && (
          <button onClick={transcribe} disabled={working}>
            Transcribe {pending} page{pending > 1 ? "s" : ""}
          </button>
        )}
        {firstToReview && (
          <button className="primary" onClick={() => go(`/page/${firstToReview.id}`)}>
            Review pages
          </button>
        )}
        <button onClick={() => runBuild(false)} disabled={building}>
          {building ? "Building…" : "Build draft PDF"}
        </button>
        <button onClick={() => runBuild(true)} disabled={building} title="No highlighting of unsure readings">
          Build final PDF
        </button>
        <button className="danger" onClick={remove}>
          Delete
        </button>
      </div>

      {build && !build.ok && (
        <div className="banner error">
          <b>The LaTeX did not compile.</b>
          {build.fixes > 0 && ` (${build.fixes} error(s) were auto-fixed first.)`}
          <ul className="errors">
            {build.errors.map((e, i) => (
              <li key={i}>
                {e.line ? `line ${e.line}: ` : ""}
                {e.message}{" "}
                {e.page_id && (
                  <a href={`#/page/${e.page_id}?block=${e.block_ids[0] ?? ""}`}>fix block{e.block_ids.length > 1 ? "s" : ""} →</a>
                )}
              </li>
            ))}
          </ul>
          {build.log_tail && <pre className="log">{build.log_tail}</pre>}
        </div>
      )}
      {build?.ok && build.fixes > 0 && <div className="banner">Built after auto-fixing {build.fixes} compile error(s).</div>}
      {build?.ok && build.errors.length > 0 && (
        <div className="banner">
          <b>Built, but {build.errors.length} block{build.errors.length > 1 ? "s" : ""} did not compile</b> and{" "}
          {build.errors.length > 1 ? "are" : "is"} shown as red raw text in the PDF:
          <ul className="errors">
            {build.errors.map((e, i) => (
              <li key={i}>
                {e.message}{" "}
                {e.page_id && <a href={`#/page/${e.page_id}?block=${e.block_ids[0] ?? ""}`}>fix →</a>}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="card">
        <h2>Pages</h2>
        <div className="thumbs">
          {doc.pages.map((p) => (
            <a key={p.id} className="thumb" href={`#/page/${p.id}`}>
              <img src={`/files/page/${p.id}.png`} alt={`Page ${p.page_index + 1}`} loading="lazy" />
              <div className="row">
                <span>Page {p.page_index + 1}</span>
                <span className={`chip ${p.status}`}>{p.status}</span>
                {p.unsure_count ? <span className="chip transcribed">{p.unsure_count} unsure</span> : null}
              </div>
              {p.error && <span className="small" style={{ color: "var(--err)" }}>{p.error}</span>}
              {p.model && <span className="small muted">{p.model}</span>}
            </a>
          ))}
        </div>
      </div>

      {pdfUrl && (
        <div className="card">
          <div className="row" style={{ marginBottom: 10 }}>
            <h2 style={{ margin: 0 }}>PDF</h2>
            <span style={{ flex: 1 }} />
            <a className="btn small" href={pdfUrl} target="_blank" rel="noreferrer">
              Open
            </a>
          </div>
          {(build?.tex_path ?? doc.tex_path) && <p className="small muted">LaTeX source: {build?.tex_path ?? doc.tex_path}</p>}
          <iframe className="pdf-frame" src={pdfUrl} title="Compiled PDF" />
        </div>
      )}
    </div>
  );
}
