// 请求封装（设计报告 §7.3）：统一 baseURL、解析 {code, message, data}、错误提示
export const BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export interface ApiEnvelope<T = unknown> {
  code: number;
  message: string;
  data: T;
}

/** 非流式 JSON 请求，返回 data 字段（code!==0 时抛错） */
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  let envelope: ApiEnvelope<T>;
  try {
    envelope = (await res.json()) as ApiEnvelope<T>;
  } catch {
    throw new Error(`网络异常（HTTP ${res.status}）`);
  }
  if (envelope.code !== 0) {
    throw new Error(envelope.message || "请求失败");
  }
  return envelope.data;
}

/** SSE 流式响应解析：逐帧 yield {delta, done, error?, data?} */
export async function* sseStream<T = Record<string, unknown>>(
  res: Response
): AsyncGenerator<T> {
  if (!res.body) throw new Error("响应无内容");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const t = line.trim();
      if (t.startsWith("data: ")) {
        try {
          yield JSON.parse(t.slice(6)) as T;
        } catch {
          // 忽略残缺帧
        }
      }
    }
  }
}

/** 选图后压缩为 jpeg blob（≤1.5MB，宽边 ≤1600px） */
export async function compressImage(file: File): Promise<Blob> {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, 1600 / Math.max(bitmap.width, bitmap.height));
  const w = Math.round(bitmap.width * scale);
  const h = Math.round(bitmap.height * scale);
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("无法处理图片");
  ctx.drawImage(bitmap, 0, 0, w, h);
  const blob = await new Promise<Blob | null>((resolve) =>
    canvas.toBlob(resolve, "image/jpeg", 0.85)
  );
  if (!blob) throw new Error("图片压缩失败");
  return blob;
}
