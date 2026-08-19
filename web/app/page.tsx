"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, BASE_URL, compressImage, sseStream } from "./lib/api";

interface Message {
  id: number | string;
  role: "user" | "assistant";
  content: string;
  image_url?: string;
  created_at?: string;
}

interface StreamFrame {
  delta?: string;
  done?: boolean;
  error?: string;
  data?: { thread_id?: string };
}

const THREAD_KEY = "sichu-thread-id";

export default function Home() {
  const [threadId, setThreadId] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [pendingImage, setPendingImage] = useState<string>(""); // 待发送图片 URL
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState("");
  const [historyLoaded, setHistoryLoaded] = useState(false);

  const scrollRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // 初始化 thread_id（localStorage 持久化，刷新后续聊）
  useEffect(() => {
    let tid = localStorage.getItem(THREAD_KEY) || "";
    if (!tid) {
      tid = crypto.randomUUID();
      localStorage.setItem(THREAD_KEY, tid);
    }
    setThreadId(tid);
  }, []);

  // 加载历史
  useEffect(() => {
    if (!threadId || historyLoaded) return;
    (async () => {
      try {
        const data = await api<{ items: Message[]; total: number }>(
          `/chat/messages?thread_id=${encodeURIComponent(threadId)}`
        );
        setMessages(data.items);
      } catch (e) {
        console.warn("历史加载失败", e);
      } finally {
        setHistoryLoaded(true);
      }
    })();
  }, [threadId, historyLoaded]);

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      scrollRef.current?.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: "smooth",
      });
    });
  }, []);

  // 发送消息（SSE 流式）
  const send = async () => {
    const text = input.trim();
    if ((!text && !pendingImage) || streaming) return;

    const userMsg: Message = {
      id: `local-${Date.now()}`,
      role: "user",
      content: text,
      image_url: pendingImage || undefined,
    };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setPendingImage("");
    setError("");
    setStreaming(true);
    scrollToBottom();

    // 占位 assistant 气泡
    const assistantId = `assistant-${Date.now()}`;
    setMessages((m) => [...m, { id: assistantId, role: "assistant", content: "" }]);

    try {
      const res = await fetch(`${BASE_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thread_id: threadId, message: text, image_url: pendingImage || undefined }),
      });
      if (!res.ok || !res.headers.get("content-type")?.includes("text/event-stream")) {
        const env = await res.json().catch(() => null);
        throw new Error(env?.message || `HTTP ${res.status}`);
      }

      let acc = "";
      for await (const frame of sseStream<StreamFrame>(res)) {
        if (frame.error) throw new Error(frame.error);
        if (frame.delta) {
          acc += frame.delta;
          setMessages((m) =>
            m.map((it) => (it.id === assistantId ? { ...it, content: acc } : it))
          );
          scrollToBottom();
        }
        if (frame.done) break;
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : "网络异常";
      setError(msg);
      // 失败时移除空占位，保留 user 消息
      setMessages((m) => m.filter((it) => it.id !== assistantId || it.content));
    } finally {
      setStreaming(false);
      setHistoryLoaded(false); // 允许下次重新拉取历史
    }
  };

  // 新对话
  const newChat = () => {
    const tid = crypto.randomUUID();
    localStorage.setItem(THREAD_KEY, tid);
    setThreadId(tid);
    setMessages([]);
    setError("");
    setPendingImage("");
    setHistoryLoaded(false);
  };

  // 清空历史
  const clearChat = async () => {
    if (!threadId) return;
    try {
      await api(`/chat/messages?thread_id=${encodeURIComponent(threadId)}`, {
        method: "DELETE",
      });
      setMessages([]);
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "清空失败");
    }
  };

  // 选图上传：OSS 签名直传（失败则回退 URL 粘贴）
  const onPickImage = async (file: File) => {
    try {
      const blob = await compressImage(file);
      const sign = await api<{
        upload_url: string;
        public_url: string;
      }>("/oss/sign", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: `dish_${Date.now()}.jpg`,
          content_type: "image/jpeg",
          size: blob.size,
        }),
      });
      const up = await fetch(sign.upload_url, {
        method: "PUT",
        headers: { "Content-Type": "image/jpeg" },
        body: blob,
      });
      if (!up.ok) throw new Error(`上传失败 HTTP ${up.status}`);
      setPendingImage(sign.public_url);
      setError("");
    } catch (e) {
      const reason = e instanceof Error ? e.message : "未知错误";
      setError(`图片上传失败（${reason}）。可直接粘贴图片 URL 到消息中。`);
    }
  };

  const enterText = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      send();
    }
  };

  return (
    <main className="app">
      {/* 背景光斑 */}
      <div className="ambient ambient-a" aria-hidden />
      <div className="ambient ambient-b" aria-hidden />

      {/* 浮岛顶栏 */}
      <header className="topbar">
        <div className="brand">
          <span className="brand-dot" />
          <span className="brand-name">私厨 AI</span>
          <span className="brand-sub">私人厨师</span>
        </div>
        <div className="topbar-actions">
          <button className="pill-btn" onClick={newChat} title="新对话">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M12 5v14M5 12h14" strokeLinecap="round" />
            </svg>
            新对话
          </button>
          <button className="pill-btn ghost" onClick={clearChat} title="清空本会话">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M3 6h18M8 6V4h8v2m1 0-1 14H8L7 6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            清空
          </button>
        </div>
      </header>

      {/* 消息区 */}
      <section className="thread" ref={scrollRef}>
        {messages.length === 0 && !streaming ? (
          <div className="welcome">
            <div className="eyebrow">拍下 · 识别 · 开做</div>
            <h1>
              把食材交给 AI，
              <br />
              今晚吃什么<em>。</em>
            </h1>
            <p className="welcome-sub">
              发送一张食材照片，私人厨师为你整理清单、搜索食谱、打分推荐。
            </p>
            <div className="hints">
              {[
                { icon: "🥕", title: "拍张食材", desc: "黄瓜·番茄·鸡蛋..." },
                { icon: "🍳", title: "生成菜谱", desc: "营养均衡度 + 难度打分" },
                { icon: "🧂", title: "缺啥买啥", desc: "补充食材清单" },
              ].map((h) => (
                <button
                  key={h.title}
                  className="hint-card"
                  onClick={() => setInput(`帮我用 ${h.title} 做一道菜`)}
                >
                  <span className="hint-icon">{h.icon}</span>
                  <span className="hint-title">{h.title}</span>
                  <span className="hint-desc">{h.desc}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="messages">
            {messages.map((m) => (
              <div key={m.id} className={`bubble-row ${m.role}`}>
                <div className={`bubble-shell ${m.role}`}>
                  <div className={`bubble ${m.role}`}>
                    {m.image_url && (
                      <a
                        className="bubble-img"
                        href={m.image_url}
                        target="_blank"
                        rel="noreferrer"
                      >
                        <img src={m.image_url} alt="食材" />
                      </a>
                    )}
                    {m.content ? (
                      <p className="bubble-text">{m.content}</p>
                    ) : (
                      m.role === "assistant" && (
                        <span className="typing">
                          <i /><i /><i />
                        </span>
                      )
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* 底部输入岛 */}
      <footer className="composer-wrap">
        <div className="composer">
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onPickImage(f);
              e.target.value = "";
            }}
          />
          {pendingImage && (
            <div className="pending-img">
              <img src={pendingImage} alt="待发送" />
              <button onClick={() => setPendingImage("")} title="移除">
                ×
              </button>
            </div>
          )}
          <div className="composer-row">
            <button className="icon-btn" onClick={() => fileRef.current?.click()} title="选择图片">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="3" y="5" width="18" height="14" rx="3" />
                <circle cx="8.5" cy="10" r="1.5" />
                <path d="m4 17 5-5 4 4 3-3 4 4" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={enterText}
              rows={1}
              placeholder={pendingImage ? "告诉厨师这些食材想怎么做？" : "描述食材，或发送图片…"}
              disabled={streaming}
            />
            <button
              className="send-btn"
              onClick={send}
              disabled={streaming || (!input.trim() && !pendingImage)}
              title="发送"
            >
              <span className="send-label">
                {streaming ? "…" : "发送"}
                <span className="send-ico">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                    <path d="M5 12h14m-6-6 6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </span>
              </span>
            </button>
          </div>
          {error && <div className="composer-err">{error}</div>}
        </div>
      </footer>
    </main>
  );
}
