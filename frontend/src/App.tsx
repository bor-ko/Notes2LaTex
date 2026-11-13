import { useEffect, useState } from "react";
import { api, Status } from "./api";
import { Library } from "./Library";
import { DocumentView } from "./DocumentView";
import { PageReview } from "./PageReview";
import { EvalView } from "./EvalView";

/** Minimal hash router: #/  #/doc/3  #/page/12?block=40  #/eval */
function useHash(): string {
  const [hash, setHash] = useState(window.location.hash || "#/");
  useEffect(() => {
    const on = () => setHash(window.location.hash || "#/");
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return hash;
}

export function go(path: string) {
  window.location.hash = path;
}

const ENGINE_KEY = "notes2tex.engine";

function savedEngine(): string | null {
  try {
    return localStorage.getItem(ENGINE_KEY);
  } catch {
    return null;
  }
}

/** The engine to use: the one picked last time if it is still available,
 * else the server default if available, else the first one that works. */
function pickEngine(status: Status, wanted: string | null): string {
  const ok = (n: string | null) => !!n && status.engines.some((e) => e.name === n && e.available);
  if (ok(wanted)) return wanted!;
  if (ok(status.default_engine)) return status.default_engine;
  return status.engines.find((e) => e.available)?.name ?? "offline";
}

export function App() {
  const hash = useHash();
  const [status, setStatus] = useState<Status | null>(null);
  const [engine, setEngineState] = useState<string>("offline");
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.status().then(
      (s) => {
        setStatus(s);
        setEngineState(pickEngine(s, savedEngine()));
      },
      (e) => setErr(String(e.message ?? e)),
    );
  }, []);

  function setEngine(name: string) {
    setEngineState(name);
    try {
      localStorage.setItem(ENGINE_KEY, name);
    } catch {
      /* private mode: just don't remember */
    }
  }

  const [path, query] = hash.slice(1).split("?");
  const parts = path.split("/").filter(Boolean);
  const params = new URLSearchParams(query ?? "");

  let view: JSX.Element;
  let wide = false;
  if (!status) view = <p className="muted">{err ? `Cannot reach the notes2tex server: ${err}` : "Loading…"}</p>;
  else if (parts[0] === "doc" && parts[1]) view = <DocumentView key={parts[1]} id={Number(parts[1])} status={status} engine={engine} />;
  else if (parts[0] === "page" && parts[1]) {
    wide = true;
    view = (
      <PageReview
        key={parts[1]}
        id={Number(parts[1])}
        status={status}
        engine={engine}
        focusBlock={params.get("block") ? Number(params.get("block")) : null}
      />
    );
  } else if (parts[0] === "eval") view = <EvalView />;
  else view = <Library status={status} engine={engine} />;

  const current = status?.engines.find((e) => e.name === engine);

  return (
    <>
      <header className="topbar">
        <a className="brand" href="#/">notes2tex</a>
        <nav>
          <a href="#/">Library</a>
          <a href="#/eval">Accuracy</a>
        </nav>
        <span className="spacer" />
        {status && (
          <label className="row small muted" title={current?.detail}>
            Engine
            <select value={engine} onChange={(e) => setEngine(e.target.value)}>
              {status.engines.map((e) => (
                <option key={e.name} value={e.name} disabled={!e.available}>
                  {e.label}
                  {e.available ? "" : " (not ready)"}
                </option>
              ))}
            </select>
          </label>
        )}
      </header>
      <main className={wide ? "wide" : ""}>{view}</main>
    </>
  );
}
