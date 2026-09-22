"use client";

import { useState } from "react";
import { ChevronRight, Loader2 } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";

/** Split "[1.4s] Searching the paper corpus: 1 paper(s) used" into its parts. */
function parseStep(step: string) {
  const m = step.match(/^\[([\d.]+s)\]\s*(.*)$/);
  if (!m) return { time: null as string | null, label: step, detail: null as string | null };
  const rest = m[2];
  const colon = rest.indexOf(": ");
  return colon === -1
    ? { time: m[1], label: rest, detail: null }
    : { time: m[1], label: rest.slice(0, colon), detail: rest.slice(colon + 2) };
}

export function ReasoningPanel({
  steps,
  streaming,
}: {
  steps: string[];
  streaming?: boolean;
}) {
  const [open, setOpen] = useState(false);
  if (!steps.length) return null;

  const last = parseStep(steps[steps.length - 1]);

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="mb-4">
      <div className="rounded-lg border border-border/60 bg-muted/30">
        <CollapsibleTrigger
          className={cn(
            "flex w-full items-center gap-2 px-3 py-2.5 text-left",
            "min-h-11 rounded-lg text-sm transition-colors hover:bg-muted/50",
          )}
        >
          <ChevronRight
            className={cn(
              "size-4 shrink-0 text-muted-foreground transition-transform",
              open && "rotate-90",
            )}
          />
          {streaming ? (
            <Loader2 className="size-3.5 shrink-0 animate-spin text-muted-foreground" />
          ) : null}
          <span className="font-medium text-foreground/90">Reasoning</span>
          <span className="text-muted-foreground">·</span>
          {/* Collapsed, the panel still shows the latest step so progress is visible. */}
          <span className="truncate text-muted-foreground">
            {open ? `${steps.length} steps` : last.label}
          </span>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <ol className="space-y-0 border-t border-border/60 px-3 py-1">
            {steps.map((step, i) => {
              const { time, label, detail } = parseStep(step);
              return (
                <li key={i} className="flex gap-3 py-1.5 text-sm">
                  <span className="w-12 shrink-0 pt-px text-right font-mono text-xs text-muted-foreground/70">
                    {time ?? ""}
                  </span>
                  <span className="min-w-0">
                    <span className="text-foreground/90">{label}</span>
                    {detail ? (
                      <span className="text-muted-foreground"> — {detail}</span>
                    ) : null}
                  </span>
                </li>
              );
            })}
          </ol>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}
