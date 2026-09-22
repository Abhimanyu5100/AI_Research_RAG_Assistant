/** Events emitted by the FastAPI backend, one JSON object per line. */
export type StreamEvent =
  | { type: "reason"; text: string }
  | { type: "token"; text: string }
  | { type: "supersede" }
  | { type: "sources"; items: string[] }
  | { type: "notice"; text: string }
  | { type: "error"; text: string };

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning?: string[];
  sources?: string[];
  notices?: string[];
  error?: string;
  /** Set while the assistant turn is still streaming. */
  pending?: boolean;
}

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
}

export const NEW_CHAT_TITLE = "New chat";
