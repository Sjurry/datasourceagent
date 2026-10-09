import 'dotenv/config';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { runAuthorityTraceAgent } from './agent.js';

const PORT = Number(process.env.TRACE_UI_PORT ?? 4399);
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url ?? '/', 'http://localhost');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }
  if (url.pathname === '/' && req.method === 'GET') {
    const html = fs.readFileSync(path.join(process.cwd(), 'trace-ui.html'), 'utf-8');
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(html); return;
  }
  if (url.pathname === '/api/trace' && req.method === 'POST') {
    let body = '';
    for await (const c of req) body += c;
    try {
      const p = JSON.parse(body || '{}');
      const result = await runAuthorityTraceAgent(
        { title: p.title ?? null, title_cn: p.title_cn ?? null, content: p.content ?? null, content_cn: p.content_cn ?? null, summary: p.summary ?? null, summary_cn: p.summary_cn ?? null },
        { taskId: `ui-${Date.now()}`, newsId: 0, timeRange: p.timeRange ?? '7d' },
      );
      res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify(result));
    } catch (e) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: e instanceof Error ? e.message : String(e) }));
    }
    return;
  }
  res.writeHead(404); res.end('not found');
});
server.listen(PORT, () => console.log(`Trace UI: http://localhost:${PORT}`));
