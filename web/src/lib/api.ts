/**
 * The backend is reached through a same-origin path that next.config rewrites
 * to FastAPI, so the browser makes no cross-origin request and no CORS
 * configuration is needed in development.
 */
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "/api";

export interface ChatRequest {
  message: string;
  session_id: string;
}

export function chatUrl() {
  return `${API_BASE}/chat`;
}

export function resetUrl() {
  return `${API_BASE}/reset`;
}

export function healthUrl() {
  return `${API_BASE}/health`;
}
