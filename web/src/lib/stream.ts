import type { StreamEvent } from "./types";

/**
 * Read a newline-delimited JSON stream.
 *
 * The backend writes one event per line, but a network chunk can split a line
 * anywhere, so bytes are buffered until a newline is seen. A line that fails to
 * parse is skipped rather than aborting the stream.
 */
export async function* readEventStream(
  body: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      if (signal?.aborted) return;
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      let newline: number;
      while ((newline = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (!line) continue;
        try {
          yield JSON.parse(line) as StreamEvent;
        } catch {
          // Partial or malformed line: drop it and keep reading.
        }
      }
    }

    // Flush a trailing line with no newline terminator.
    const tail = buffer.trim();
    if (tail) {
      try {
        yield JSON.parse(tail) as StreamEvent;
      } catch {
        /* ignore */
      }
    }
  } finally {
    reader.releaseLock();
  }
}
