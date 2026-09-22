"use client";

import { memo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";

/**
 * Models emit \( inline \) and \[ display \] delimiters, which remark-math does
 * not recognise; it expects $ and $$. Rewriting them is safe mid-stream because
 * each pattern requires its closing delimiter, so a half-arrived equation is
 * left alone until it completes.
 */
const DISPLAY = /\\\[([\s\S]+?)\\\]/g;
const INLINE = /\\\(([\s\S]+?)\\\)/g;
/** Built-in browsing models emit internal reference markers; strip them. */
const CITE_ARTIFACT = /【[^】]{0,40}】/g;

export function normalizeMath(text: string): string {
  return text
    .replace(CITE_ARTIFACT, "")
    .replace(DISPLAY, (_, body) => `\n\n$$\n${String(body).trim()}\n$$\n\n`)
    .replace(INLINE, (_, body) => `$${String(body).trim()}$`);
}

function MarkdownImpl({ children }: { children: string }) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[
          [rehypeKatex, { throwOnError: false, strict: false }],
          [rehypeHighlight, { detect: true, ignoreMissing: true }],
        ]}
        components={{
          // Wide tables scroll instead of compressing cells to one word each.
          table: ({ children }) => (
            <div className="md-table-wrap">
              <table>{children}</table>
            </div>
          ),
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noopener noreferrer">
              {children}
            </a>
          ),
        }}
      >
        {normalizeMath(children)}
      </ReactMarkdown>
    </div>
  );
}

export const Markdown = memo(MarkdownImpl);
