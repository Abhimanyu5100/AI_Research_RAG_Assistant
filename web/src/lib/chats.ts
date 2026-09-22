"use client";

import type { Chat } from "./types";
import { NEW_CHAT_TITLE } from "./types";

const KEY = "rag-assistant.chats.v1";

/**
 * Chats are persisted in localStorage so a refresh does not lose the
 * transcript. The backend keeps its own conversation memory keyed by the same
 * chat id, so the two stay aligned.
 */
export function loadChats(): Chat[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as Chat[]) : [];
  } catch {
    return [];
  }
}

export function saveChats(chats: Chat[]) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(KEY, JSON.stringify(chats));
  } catch {
    // Quota or private mode: the app still works for this session.
  }
}

export function newChat(): Chat {
  return {
    id: crypto.randomUUID(),
    title: NEW_CHAT_TITLE,
    messages: [],
    createdAt: Date.now(),
  };
}

/** Derive a readable chat title from the first thing the user asked. */
export function titleFrom(text: string): string {
  const clean = text.replace(/\s+/g, " ").trim();
  if (clean.length <= 48) return clean;
  const cut = clean.slice(0, 48);
  const lastSpace = cut.lastIndexOf(" ");
  return (lastSpace > 24 ? cut.slice(0, lastSpace) : cut) + "…";
}
