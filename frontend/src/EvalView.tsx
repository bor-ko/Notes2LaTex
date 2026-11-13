import { useEffect, useState } from "react";
import { api, EvalReport, ScoreRow } from "./api";

const TARGET = 0.97;
const pct = (x: number | null, digits = 1) => (x === null ? "–" : `${(x * 100).toFixed(digits)}%`);

function Table({ title, rows }: { title: string; rows: Record<string, ScoreRow> }) {
  const entries = Object.entries(rows);
  return (
    <div className="card">
      <h2>By {title}</h2>
      {entries.length === 0 ? (
        <p className="muted small">No data yet.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>{title}</th>
              <th>Token acc.</th>
              <th>Exact</th>
              <th>Blocks</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([name, s]) => (
              <tr key={name}>
                <td>{name}</td>
                <td style={{ color: s.token_accuracy !== null && s.token_accuracy >= TARGET ? "var(--ok)" : undefined }}>
                  {pct(s.token_accuracy)}
                </td>
                <td>{pct(s.exact_match, 0)}</td>
                <td>{s.blocks}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export function EvalView() {
  const [split, setSplit] = useState("test");
  const [rep, setRep] = useState<EvalReport | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    setRep(null);
    api.evalReport(split).then(setRep, (e) => setErr(e.message));
  }, [split]);

  if (err) return <p className="banner error">{err}</p>;

  const o = rep?.overall;
  return (
    <div className="stack">
      <div className="row">
        <h1>Accuracy</h1>
        <span style={{ flex: 1 }} />
        <label className="row small muted">
          Split
          <select value={split} onChange={(e) => setSplit(e.target.value)}>
            <option value="test">test (held out)</option>
            <option value="val">validation</option>
            <option value="train">train</option>
            <option value="all">all</option>
          </select>
        </label>
      </div>
      <div className="card">
        {!rep ? (
          <p className="muted">Loading…</p>
        ) : !o || !o.blocks ? (
          <p className="muted">
            No approved pages in the <b>{split}</b> split yet. Each document is randomly assigned to train / validation / test when it
            is imported; approve pages and their accuracy shows up here. Documents per split so far: train{" "}
            {rep.documents_per_split.train}, val {rep.documents_per_split.val}, test {rep.documents_per_split.test}.
          </p>
        ) : (
          <div className="row" style={{ gap: 40, alignItems: "flex-start" }}>
            <div>
              <div className="big-number" style={{ color: (o.token_accuracy ?? 0) >= TARGET ? "var(--ok)" : undefined }}>
                {pct(o.token_accuracy, 2)}
              </div>
              <div className="target muted">token accuracy · target {pct(TARGET, 0)}</div>
            </div>
            <div className="stack" style={{ gap: 4 }}>
              <span>
                Exact blocks: <b>{pct(o.exact_match)}</b>
              </span>
              <span>
                Role correct: <b>{pct(o.role_accuracy)}</b>
              </span>
              <span className="muted small">
                {o.blocks} blocks · {o.ref_tokens} LaTeX tokens · {rep.reviewed_pages} reviewed pages overall
              </span>
            </div>
          </div>
        )}
        <p className="small muted" style={{ marginTop: 14, marginBottom: 0 }}>
          Token accuracy = 1 − (edit distance between the model's draft and your approved LaTeX ÷ length of yours), after ignoring
          differences that do not change the rendered result (spacing, <code>\dfrac</code> vs <code>\frac</code>, <code>^{"{2}"}</code> vs{" "}
          <code>^2</code>, <code>\left(</code>…). Use the <b>test</b> split for the honest number: those documents are never used for
          training.
        </p>
      </div>
      {rep && o && o.blocks > 0 && (
        <div className="grid-3">
          <Table title="subject" rows={rep.subject} />
          <Table title="role" rows={rep.role} />
          <Table title="model" rows={rep.model} />
        </div>
      )}
    </div>
  );
}
