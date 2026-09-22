"use client";

import { BookOpen, GitCompare, Globe } from "lucide-react";

const EXAMPLES = [
  {
    icon: BookOpen,
    label: "What is reinforcement learning?",
    hint: "Answered from the indexed papers",
  },
  {
    icon: GitCompare,
    label: "Compare Q-learning with policy gradient methods",
    hint: "Retrieval plus synthesis",
  },
  {
    icon: Globe,
    label: "Explain multi-agent coordination challenges, and who won the 2024 F1 championship?",
    hint: "Split across the corpus and web search",
  },
];

export function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col justify-center px-4 py-12">
      <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
        Research assistant
      </h1>
      <p className="mt-2 max-w-xl text-[15px] leading-7 text-muted-foreground">
        Ask about AI and machine learning and it searches an indexed arXiv corpus,
        citing the papers it used. Other questions are answered with web search.
      </p>

      <div className="mt-8 grid gap-2">
        {EXAMPLES.map(({ icon: Icon, label, hint }) => (
          <button
            key={label}
            onClick={() => onPick(label)}
            className="group flex min-h-11 items-start gap-3 rounded-xl border border-border/70 bg-card/50 px-4 py-3 text-left transition-colors hover:border-border hover:bg-card"
          >
            <Icon className="mt-0.5 size-4 shrink-0 text-muted-foreground transition-colors group-hover:text-primary" />
            <span className="min-w-0">
              <span className="block text-sm leading-6 text-foreground">{label}</span>
              <span className="block text-xs text-muted-foreground">{hint}</span>
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
