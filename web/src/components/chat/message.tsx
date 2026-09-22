"use client";

import { AlertTriangle, FileText, Sparkles, TriangleAlert, User } from "lucide-react";
import type { Message } from "@/lib/types";
import { Markdown } from "./markdown";
import { ReasoningPanel } from "./reasoning";

function Sources({ items }: { items: string[] }) {
  if (!items.length) return null;
  return (
    <div className="mt-5 flex flex-wrap items-center gap-1.5 border-t border-border/60 pt-3">
      <span className="mr-1 text-xs font-medium text-muted-foreground">Sources</span>
      {items.map((s) => (
        <span
          key={s}
          className="inline-flex items-center gap-1.5 rounded-md border border-border/70 bg-muted/40 px-2 py-1 font-mono text-xs text-foreground/80"
        >
          <FileText className="size-3 text-muted-foreground" />
          {s}
        </span>
      ))}
    </div>
  );
}

function Notice({ text, tone = "warn" }: { text: string; tone?: "warn" | "error" }) {
  const Icon = tone === "error" ? AlertTriangle : TriangleAlert;
  return (
    <div
      className={
        tone === "error"
          ? "mb-4 flex gap-2.5 rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2.5 text-sm text-foreground"
          : "mb-4 flex gap-2.5 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2.5 text-sm text-foreground"
      }
    >
      <Icon
        className={
          tone === "error"
            ? "mt-0.5 size-4 shrink-0 text-destructive"
            : "mt-0.5 size-4 shrink-0 text-amber-500"
        }
      />
      <span className="min-w-0 whitespace-pre-wrap">{text}</span>
    </div>
  );
}

function Caret() {
  return (
    <span className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 animate-pulse rounded-full bg-foreground/70 align-middle" />
  );
}

export function MessageView({ message }: { message: Message }) {
  const isUser = message.role === "user";

  return (
    <article className="flex gap-3 sm:gap-4">
      <div
        aria-hidden
        className={
          isUser
            ? "mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground"
            : "mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-primary/15 text-primary"
        }
      >
        {isUser ? <User className="size-3.5" /> : <Sparkles className="size-3.5" />}
      </div>

      <div className="min-w-0 flex-1">
        <div className="mb-1.5 text-xs font-medium text-muted-foreground">
          {isUser ? "You" : "Assistant"}
        </div>

        {isUser ? (
          <p className="whitespace-pre-wrap text-[15px] leading-7 text-foreground">
            {message.content}
          </p>
        ) : (
          <>
            {message.reasoning?.length ? (
              <ReasoningPanel steps={message.reasoning} streaming={message.pending} />
            ) : null}
            {message.notices?.map((n, i) => (
              <Notice key={i} text={n} />
            ))}
            {message.error ? <Notice text={message.error} tone="error" /> : null}
            {message.content ? (
              <Markdown>{message.content}</Markdown>
            ) : message.pending && !message.reasoning?.length ? (
              <div className="text-sm text-muted-foreground">Thinking…</div>
            ) : null}
            {message.pending && message.content ? <Caret /> : null}
            <Sources items={message.sources ?? []} />
          </>
        )}
      </div>
    </article>
  );
}
