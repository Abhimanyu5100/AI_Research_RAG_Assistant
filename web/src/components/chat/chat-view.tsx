"use client";

import { useEffect, useRef } from "react";
import type { Message } from "@/lib/types";
import { MessageView } from "./message";
import { EmptyState } from "./empty-state";

export function ChatView({
  messages,
  onPickExample,
}: {
  messages: Message[];
  onPickExample: (q: string) => void;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);

  // Follow the stream, but stop following the moment the user scrolls up.
  useEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;
    const onScroll = () => {
      const gap = el.scrollHeight - el.scrollTop - el.clientHeight;
      pinned.current = gap < 120;
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  const last = messages[messages.length - 1];
  useEffect(() => {
    if (pinned.current) bottomRef.current?.scrollIntoView({ block: "end" });
  }, [messages.length, last?.content, last?.reasoning?.length]);

  if (!messages.length) {
    return (
      <div ref={scrollerRef} className="flex flex-1 overflow-y-auto">
        <EmptyState onPick={onPickExample} />
      </div>
    );
  }

  return (
    <div ref={scrollerRef} className="flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-3xl space-y-8 px-4 py-8">
        {messages.map((m) => (
          <MessageView key={m.id} message={m} />
        ))}
        <div ref={bottomRef} className="h-px" />
      </div>
    </div>
  );
}
