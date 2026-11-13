export type Subject = { key: string; name: string; examples: number; default_code_language: string | null };
export type Engine = { name: string; label: string; available: boolean; detail: string };
export type Status = { default_engine: string; engines: Engine[]; roles: string[]; subjects: Subject[] };

export type DocSummary = {
  id: number;
  name: string;
  subject: string;
  split: string;
  created_at: string;
  page_count: number;
  reviewed_count: number | null;
  transcribed_count: number | null;
  pending_count: number | null;
  error_count: number | null;
};

export type PageSummary = {
  id: number;
  page_index: number;
  status: "new" | "transcribing" | "transcribed" | "reviewed" | "error";
  model: string | null;
  error: string | null;
  block_count: number;
  unsure_count: number | null;
  cost: number | null;
};

export type DocDetail = DocSummary & { pages: PageSummary[]; pdf_url: string | null; tex_path: string | null };

export type Block = {
  id: number;
  ord: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  role: string;
  latex: string;
  draft_role: string | null;
  draft_latex: string | null;
  final_latex: string | null;
  continues_previous: number | boolean;
  code_language: string | null;
  crop_url: string;
  unsure: boolean;
};

export type PageDetail = {
  id: number;
  page_index: number;
  width: number;
  height: number;
  status: PageSummary["status"];
  model: string | null;
  error: string | null;
  document: { id: number; name: string; subject: string; split: string };
  image_url: string;
  prev_id: number | null;
  next_id: number | null;
  page_count: number;
  blocks: Block[];
};

export type BlockEdit = { id: number; role: string; latex: string; continues_previous: boolean; code_language: string | null };

export type BuildError = { message: string; line: number | null; page_id: number | null; block_ids: number[] };
export type BuildResult = { ok: boolean; fixes: number; errors: BuildError[]; log_tail: string; pdf_url: string | null; tex_path: string | null };

export type ScoreRow = { blocks: number; ref_tokens: number; token_accuracy: number | null; exact_match: number | null; role_accuracy: number | null };
export type EvalReport = {
  split: string;
  documents_per_split: Record<string, number>;
  reviewed_pages: number;
  overall: ScoreRow;
  subject: Record<string, ScoreRow>;
  role: Record<string, ScoreRow>;
  model: Record<string, ScoreRow>;
};

async function req<T>(method: string, url: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers = { "Content-Type": "application/json" };
  }
  const res = await fetch(url, init);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (data.detail) msg = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch {
      /* not json */
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  status: () => req<Status>("GET", "/api/status"),
  documents: () => req<DocSummary[]>("GET", "/api/documents"),
  upload: (form: FormData) => req<{ id: number }>("POST", "/api/documents", form),
  document: (id: number) => req<DocDetail>("GET", `/api/documents/${id}`),
  deleteDocument: (id: number) => req<{ ok: boolean }>("DELETE", `/api/documents/${id}`),
  transcribeDocument: (id: number, engine: string, all = false) =>
    req<{ queued: number }>("POST", `/api/documents/${id}/transcribe`, { engine, all }),
  build: (id: number, final: boolean, engine: string) => req<BuildResult>("POST", `/api/documents/${id}/build`, { final, engine }),
  page: (id: number) => req<PageDetail>("GET", `/api/pages/${id}`),
  saveBlocks: (id: number, edits: BlockEdit[]) => req<PageDetail>("PUT", `/api/pages/${id}/blocks`, edits),
  approve: (id: number, edits: BlockEdit[]) => req<PageDetail>("POST", `/api/pages/${id}/approve`, edits),
  retranscribe: (id: number, engine: string) => req<PageDetail>("POST", `/api/pages/${id}/transcribe`, { engine }),
  mergeNext: (blockId: number) => req<PageDetail>("POST", `/api/blocks/${blockId}/merge-next`),
  split: (blockId: number, at: number) => req<PageDetail>("POST", `/api/blocks/${blockId}/split`, { at }),
  evalReport: (split: string) => req<EvalReport>("GET", `/api/eval?split=${encodeURIComponent(split)}`),
};

export const toEdit = (b: Block): BlockEdit => ({
  id: b.id,
  role: b.role,
  latex: b.latex ?? "",
  continues_previous: Boolean(b.continues_previous),
  code_language: b.code_language,
});
