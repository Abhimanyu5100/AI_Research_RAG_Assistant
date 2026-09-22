"use client";

import { useEffect, useRef } from "react";
import { ArrowUp, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function Composer({
  value,
  onChange,
  onSubmit,
  onStop,
  busy,
}: {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  busy: boolean;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);

  // Grow with the content up to a ceiling, then scroll.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  return (
    <div className="border-t border-border/60 bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="mx-auto w-full max-w-3xl px-4 py-4">
        <div
          className={cn(
            "flex items-end gap-2 rounded-2xl border border-border bg-card px-3 py-2.5",
            "shadow-sm transition-colors focus-within:border-ring/60",
          )}
        >
          <textarea
            ref={ref}
            rows={1}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={(e) => {
              // Enter sends; Shift+Enter inserts a newline.
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (!busy && value.trim()) onSubmit();
              }
            }}
            placeholder="Ask about AI/ML research, or anything else…"
            aria-label="Message"
            className="max-h-[200px] min-h-6 flex-1 resize-none bg-transparent py-1.5 text-[15px] leading-6 outline-none placeholder:text-muted-foreground/70"
          />
          {busy ? (
            <Button
              size="icon"
              variant="secondary"
              onClick={onStop}
              aria-label="Stop generating"
              className="size-9 shrink-0 rounded-xl"
            >
              <Square className="size-3.5 fill-current" />
            </Button>
          ) : (
            <Button
              size="icon"
              onClick={onSubmit}
              disabled={!value.trim()}
              aria-label="Send message"
              className="size-9 shrink-0 rounded-xl"
            >
              <ArrowUp className="size-4" />
            </Button>
          )}
        </div>
        <p className="mt-2 px-1 text-xs text-muted-foreground/70">
          Retrieval runs over the indexed arXiv corpus. Answers can be wrong — check the sources.
        </p>
      </div>
    </div>
  );
}
