import { FormEvent, useEffect, useState } from "react";
import { api, DocSummary, Status } from "./api";
import { go } from "./App";

export function Library({ status, engine }: { status: Status; engine: string }) {
  const [docs, setDocs] = useState<DocSummary[] | null>(null);
  const [subject, setSubject] = useState(status.subjects[0]?.key ?? "");
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.documents().then(setDocs, (e) => setErr(e.message));
  }, []);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setErr(null);
    const form = new FormData();
    form.append("file", file);
    form.append("subject", subject);
    if (name.trim()) form.append("name", name.trim());
    form.append("engine", engine);
    try {
      const { id } = await api.upload(form);
      go(`/doc/${id}`);
    } catch (e) {
      setErr((e as Error).message);
      setBusy(false);
    }
  }

  return (
    <div className="stack">
      {status.engines
        .filter((e) => !e.available && e.name === status.default_engine)
        .map((e) => (
          <div key={e.name} className="banner">
            <b>{e.label}</b> is not ready: {e.detail} Pages will use the engine picked at the top right until then.
          </div>
        ))}
      <form className="card" onSubmit={submit}>
        <h2>Add notes</h2>
        <div className="upload">
          <label>
            Tablet export (PDF or image)
            <input type="file" accept=".pdf,image/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </label>
          <label>
            Subject
            <select value={subject} onChange={(e) => setSubject(e.target.value)}>
              {status.subjects.map((s) => (
                <option key={s.key} value={s.key}>
                  {s.name}
                  {s.examples ? ` (${s.examples} examples)` : ""}
                </option>
              ))}
            </select>
          </label>
          <label>
            Title (optional)
            <input type="text" placeholder="e.g. Lecture 4: Oscillations" value={name} onChange={(e) => setName(e.target.value)} />
          </label>
          <button className="primary" disabled={!file || busy}>
            {busy ? "Importing…" : "Import & transcribe"}
          </button>
        </div>
        <p className="small muted" style={{ marginTop: 10, marginBottom: 0 }}>
          Transcribed with: <b>{status.engines.find((e) => e.name === engine)?.label ?? engine}</b> (change it at the top right).
        </p>
        {err && <p className="banner error">{err}</p>}
      </form>

      <div className="card">
        <h2>Notes</h2>
        {docs === null ? (
          <p className="muted">Loading…</p>
        ) : docs.length === 0 ? (
          <p className="muted">Nothing imported yet. Export a few pages from your tablet app as PDF and add them above.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Title</th>
                <th>Subject</th>
                <th>Pages</th>
                <th>Reviewed</th>
                <th>Split</th>
                <th>Added</th>
              </tr>
            </thead>
            <tbody>
              {docs.map((d) => {
                const reviewed = d.reviewed_count ?? 0;
                return (
                  <tr key={d.id} className="clickable" onClick={() => go(`/doc/${d.id}`)}>
                    <td>{d.name}</td>
                    <td>{status.subjects.find((s) => s.key === d.subject)?.name ?? d.subject}</td>
                    <td>
                      {d.page_count}
                      {d.pending_count ? <span className="chip transcribing" style={{ marginLeft: 6 }}>working</span> : null}
                      {d.error_count ? <span className="chip error" style={{ marginLeft: 6 }}>{d.error_count} failed</span> : null}
                    </td>
                    <td>
                      <div className="row">
                        <div className="progress">
                          <div style={{ width: `${d.page_count ? (100 * reviewed) / d.page_count : 0}%` }} />
                        </div>
                        <span className="small muted">
                          {reviewed}/{d.page_count}
                        </span>
                      </div>
                    </td>
                    <td>
                      <span className="chip split" title="Test documents are held out to measure accuracy honestly.">
                        {d.split}
                      </span>
                    </td>
                    <td className="small muted">{d.created_at.slice(0, 10)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
