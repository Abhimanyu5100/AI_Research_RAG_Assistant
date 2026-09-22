# Frontend

Next.js interface for the AI Research RAG Assistant. It talks to the FastAPI
backend over a newline-delimited JSON stream and renders reasoning steps,
answers, maths and citations as they arrive.

```bash
npm install
npm run dev
```

Opens on <http://localhost:3000> and proxies `/api/*` to the backend at
`http://127.0.0.1:8000`. Point it elsewhere with `BACKEND_ORIGIN`:

```bash
BACKEND_ORIGIN=http://localhost:9000 npm run dev
```

To run the backend and this together, use `../scripts/dev.sh`.

## Layout

| Path | Purpose |
|---|---|
| `src/app/page.tsx` | State, streaming, session wiring |
| `src/app/globals.css` | Theme tokens and rendered-answer typography |
| `src/components/chat/` | Chat view, message, reasoning panel, composer, sidebar |
| `src/lib/stream.ts` | NDJSON reader, tolerant of chunk boundaries |
| `src/lib/chats.ts` | Transcript persistence in `localStorage` |

See the [project README](../README.md) for the streaming protocol and backend
setup.
