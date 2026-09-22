"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Menu, PanelLeftClose } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { ChatView } from "@/components/chat/chat-view";
import { Composer } from "@/components/chat/composer";
import { Sidebar } from "@/components/chat/sidebar";
import { ThemeToggle } from "@/components/theme-toggle";
import { chatUrl, resetUrl } from "@/lib/api";
import { loadChats, newChat, saveChats, titleFrom } from "@/lib/chats";
import { readEventStream } from "@/lib/stream";
import type { Chat, Message } from "@/lib/types";
import { NEW_CHAT_TITLE } from "@/lib/types";

export default function Page() {
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // Restore from localStorage once on the client. This has to happen in an
  // effect rather than a lazy initialiser: the server prerender cannot read
  // localStorage, so seeding state during render would mismatch on hydration.
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    const stored = loadChats();
    const restored = stored.length ? stored : [newChat()];
    setChats(restored);
    setActiveId(restored[0].id);
    setHydrated(true);
  }, []);
  /* eslint-enable react-hooks/set-state-in-effect */

  useEffect(() => {
    if (hydrated) saveChats(chats);
  }, [chats, hydrated]);

  const active = chats.find((c) => c.id === activeId) ?? null;

  const patchActive = useCallback(
    (fn: (c: Chat) => Chat) =>
      setChats((prev) => prev.map((c) => (c.id === activeId ? fn(c) : c))),
    [activeId],
  );

  const handleCreate = useCallback(() => {
    const c = newChat();
    setChats((prev) => [c, ...prev]);
    setActiveId(c.id);
    setSidebarOpen(false);
  }, []);

  const handleDelete = useCallback(
    (id: string) => {
      // Clear the backend's memory for this session too, not just the transcript.
      void fetch(resetUrl(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "", session_id: id }),
      }).catch(() => undefined);

      setChats((prev) => {
        const next = prev.filter((c) => c.id !== id);
        if (next.length === 0) {
          const c = newChat();
          setActiveId(c.id);
          return [c];
        }
        if (id === activeId) setActiveId(next[0].id);
        return next;
      });
    },
    [activeId],
  );

  const send = useCallback(
    async (text: string) => {
      const question = text.trim();
      if (!question || !activeId || busy) return;

      const userMsg: Message = { id: crypto.randomUUID(), role: "user", content: question };
      const botId = crypto.randomUUID();
      const botMsg: Message = {
        id: botId,
        role: "assistant",
        content: "",
        reasoning: [],
        sources: [],
        notices: [],
        pending: true,
      };

      patchActive((c) => ({
        ...c,
        title: c.title === NEW_CHAT_TITLE ? titleFrom(question) : c.title,
        messages: [...c.messages, userMsg, botMsg],
      }));
      setInput("");
      setBusy(true);

      const controller = new AbortController();
      abortRef.current = controller;

      const update = (fn: (m: Message) => Message) =>
        patchActive((c) => ({
          ...c,
          messages: c.messages.map((m) => (m.id === botId ? fn(m) : m)),
        }));

      try {
        const res = await fetch(chatUrl(), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: question, session_id: activeId }),
          signal: controller.signal,
        });

        if (!res.ok || !res.body) {
          throw new Error(`Backend returned ${res.status}`);
        }

        for await (const ev of readEventStream(res.body, controller.signal)) {
          switch (ev.type) {
            case "token":
              update((m) => ({ ...m, content: m.content + ev.text }));
              break;
            case "supersede":
              // The critique asked for a revision: drop the superseded draft.
              update((m) => ({ ...m, content: "" }));
              break;
            case "reason":
              update((m) => ({ ...m, reasoning: [...(m.reasoning ?? []), ev.text] }));
              break;
            case "sources":
              update((m) => ({ ...m, sources: ev.items }));
              break;
            case "notice":
              update((m) => ({ ...m, notices: [...(m.notices ?? []), ev.text] }));
              break;
            case "error":
              update((m) => ({ ...m, error: ev.text }));
              break;
          }
        }
      } catch (err) {
        if ((err as Error)?.name === "AbortError") {
          update((m) => ({ ...m, notices: [...(m.notices ?? []), "Stopped."] }));
        } else {
          const detail = err instanceof Error ? err.message : String(err);
          update((m) => ({
            ...m,
            error: `Could not reach the backend (${detail}). Is it running on port 8000?`,
          }));
          toast.error("Backend unreachable", {
            description: "Start it with: uvicorn src.api.main:app --port 8000",
          });
        }
      } finally {
        update((m) => ({ ...m, pending: false }));
        setBusy(false);
        abortRef.current = null;
      }
    },
    [activeId, busy, patchActive],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  return (
    <div className="flex h-dvh overflow-hidden bg-background">
      {/* Desktop sidebar */}
      <aside className="hidden w-64 shrink-0 border-r border-border/60 md:block">
        <Sidebar
          chats={chats}
          activeId={activeId}
          onSelect={setActiveId}
          onCreate={handleCreate}
          onDelete={handleDelete}
        />
      </aside>

      {/* Mobile drawer */}
      {sidebarOpen ? (
        <div className="fixed inset-0 z-40 md:hidden">
          <button
            aria-label="Close sidebar"
            className="absolute inset-0 bg-black/50"
            onClick={() => setSidebarOpen(false)}
          />
          <aside className="absolute inset-y-0 left-0 w-72 border-r border-border/60 shadow-xl">
            <Sidebar
              chats={chats}
              activeId={activeId}
              onSelect={(id) => {
                setActiveId(id);
                setSidebarOpen(false);
              }}
              onCreate={handleCreate}
              onDelete={handleDelete}
              onClose={() => setSidebarOpen(false)}
            />
          </aside>
        </div>
      ) : null}

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center gap-2 border-b border-border/60 px-3 sm:px-4">
          <Button
            variant="ghost"
            size="icon"
            className="size-10 md:hidden"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open sidebar"
          >
            <Menu className="size-4" />
          </Button>
          <PanelLeftClose className="hidden size-4 text-muted-foreground md:block" />
          <span className="truncate text-sm font-medium">
            {active?.title === NEW_CHAT_TITLE ? "Research assistant" : active?.title}
          </span>
          <div className="ml-auto">
            <ThemeToggle />
          </div>
        </header>

        <ChatView messages={active?.messages ?? []} onPickExample={(q) => void send(q)} />

        <Composer
          value={input}
          onChange={setInput}
          onSubmit={() => void send(input)}
          onStop={stop}
          busy={busy}
        />
      </main>
    </div>
  );
}
