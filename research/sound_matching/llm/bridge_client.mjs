#!/usr/bin/env node
// One bounded local HarnessBridge request. No credentials or upstream errors are printed.
import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';

const REQUEST_BYTES = 262144;
const RESPONSE_BYTES = 1048576;
const MAX_BYTES = 16777216;

class ClientError extends Error {
  constructor(code, status) { super(code); this.code = code; this.status = status; }
}

function object(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function limit(value, fallback, maximum) {
  if (value === undefined) return fallback;
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0 || number > maximum) {
    throw new ClientError('configuration');
  }
  return number;
}

async function readFileBounded(filename, maximum) {
  let file;
  try {
    const before = await fs.lstat(filename);
    if (!before.isFile() || before.isSymbolicLink()) throw new ClientError('configuration');
    file = await fs.open(filename, 'r');
    const stat = await file.stat();
    if (!stat.isFile() || stat.size > maximum) throw new ClientError('configuration');
    const buffer = Buffer.alloc(maximum + 1);
    const { bytesRead } = await file.read(buffer, 0, maximum + 1, 0);
    if (bytesRead > maximum) throw new ClientError('configuration');
    return new TextDecoder('utf-8', { fatal: true }).decode(buffer.subarray(0, bytesRead));
  } catch {
    throw new ClientError('configuration');
  } finally {
    await file?.close();
  }
}

function parseArguments(argv) {
  const options = { config: null, mode: 'complete', timeout: undefined };
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === '--config' && options.config === null && argv[index + 1]) {
      options.config = argv[++index];
    } else if (value === '--timeout-seconds' && options.timeout === undefined && argv[index + 1]) {
      options.timeout = limit(argv[++index], 120, 600) * 1000;
    } else if ((value === '--health' || value === '--models') && options.mode === 'complete') {
      options.mode = value.slice(2);
    } else {
      throw new ClientError('arguments');
    }
  }
  if (!options.config || options.config.includes('\0')) throw new ClientError('arguments');
  return options;
}

function parseRequest(value) {
  if (!object(value) || Object.keys(value).some(key => !['model', 'messages', 'stream'].includes(key)) || value.stream !== false) {
    throw new ClientError('request');
  }
  if (typeof value.model !== 'string' || !value.model || value.model.length > 128 || !Array.isArray(value.messages) || value.messages.length < 1 || value.messages.length > 128) {
    throw new ClientError('request');
  }
  let dialogue = false;
  const messages = value.messages.map(message => {
    if (!object(message) || Object.keys(message).length !== 2 || !['system', 'developer', 'user', 'assistant'].includes(message.role) || typeof message.content !== 'string' || message.content.length > 100000) {
      throw new ClientError('request');
    }
    if (!Object.keys(message).every(key => key === 'role' || key === 'content')) throw new ClientError('request');
    if (message.role === 'system' || message.role === 'developer') {
      if (dialogue) throw new ClientError('request');
    } else dialogue = true;
    return { role: message.role, content: message.content };
  });
  if (messages.at(-1).role !== 'user' || !messages.at(-1).content.trim()) throw new ClientError('request');
  return { model: value.model, messages, stream: false };
}

function readInput(maximum, signal) {
  return new Promise((resolve, reject) => {
    let bytes = 0;
    const chunks = [];
    const cleanup = () => {
      process.stdin.off('data', data); process.stdin.off('end', end); process.stdin.off('error', error);
      signal.removeEventListener('abort', abort);
    };
    const error = () => { cleanup(); reject(new ClientError('request')); };
    const abort = () => { cleanup(); process.stdin.destroy(); reject(new ClientError('timeout')); };
    const data = chunk => {
      bytes += chunk.length;
      if (bytes > maximum) { cleanup(); process.stdin.destroy(); reject(new ClientError('request-limit')); return; }
      chunks.push(chunk);
    };
    const end = () => {
      cleanup();
      try { resolve(parseRequest(JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks))))); }
      catch { reject(new ClientError('request')); }
    };
    if (signal.aborted) { abort(); return; }
    process.stdin.on('data', data); process.stdin.once('end', end); process.stdin.once('error', error);
    signal.addEventListener('abort', abort, { once: true });
  });
}

function request(host, port, token, route, body, maximum, signal) {
  return new Promise((resolve, reject) => {
    const hostHeader = `${host === '::1' ? '[::1]' : host}:${port}`;
    const headers = { Host: hostHeader, Authorization: `Bearer ${token}`, Accept: 'application/json', Connection: 'close' };
    if (body !== null) {
      headers['Content-Type'] = 'application/json';
      headers['Content-Length'] = body.length;
    }
    const req = http.request({ hostname: host, port, path: route, method: body === null ? 'GET' : 'POST', headers, agent: false, maxHeaderSize: 8192, signal }, res => {
      if (res.statusCode !== 200) {
        res.destroy(); reject(new ClientError('http-status', res.statusCode)); return;
      }
      if (res.headers['content-encoding'] && res.headers['content-encoding'] !== 'identity') {
        res.destroy(); reject(new ClientError('response')); return;
      }
      const length = res.headers['content-length'];
      if (length !== undefined && (!/^\d+$/.test(length) || Number(length) > maximum)) {
        res.destroy(); reject(new ClientError('response-limit')); return;
      }
      let bytes = 0;
      const chunks = [];
      res.on('data', chunk => {
        bytes += chunk.length;
        if (bytes > maximum) { res.destroy(); reject(new ClientError('response-limit')); return; }
        chunks.push(chunk);
      });
      res.once('end', () => {
        try {
          const result = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks)));
          if (!object(result)) throw new Error();
          resolve(result);
        } catch { reject(new ClientError('response')); }
      });
      res.once('error', () => reject(new ClientError(signal.aborted ? 'timeout' : 'connection')));
      res.once('aborted', () => reject(new ClientError(signal.aborted ? 'timeout' : 'connection')));
    });
    req.once('error', () => reject(new ClientError(signal.aborted ? 'timeout' : 'connection')));
    req.end(body);
  });
}

function metadata(value, maximum) {
  if (value === undefined || value === null) return null;
  if (typeof value !== 'string' || !value || value.length > maximum) throw new ClientError('response');
  return value;
}

function normalize(document) {
  if (!Array.isArray(document.choices) || document.choices.length !== 1 || !object(document.choices[0])) throw new ClientError('response');
  const choice = document.choices[0];
  const message = choice.message;
  if (choice.finish_reason !== 'stop' || !object(message) || Object.keys(message).length !== 2 || message.role !== 'assistant' || typeof message.content !== 'string') throw new ClientError('response');
  let usage = null;
  if (document.usage !== undefined && document.usage !== null) {
    const keys = object(document.usage) ? Object.keys(document.usage) : [];
    if (!keys.length || keys.some(key => !['prompt_tokens', 'completion_tokens', 'total_tokens'].includes(key)) || Object.values(document.usage).some(value => !Number.isSafeInteger(value) || value < 0)) throw new ClientError('response');
    usage = document.usage;
  }
  return { content: message.content, usage, response_id: metadata(document.id, 256), response_model: metadata(document.model, 128) };
}

function readiness(document, mode) {
  if (mode === 'health') {
    if (document.status !== 'ok' || typeof document.version !== 'string' || document.version.length > 64 || !Number.isSafeInteger(document.active_requests) || document.active_requests < 0) throw new ClientError('response');
    return { status: 'ok', version: document.version, active_requests: document.active_requests };
  }
  if (document.object !== 'list' || !Array.isArray(document.data) || document.data.length > 128) throw new ClientError('response');
  return { object: 'list', data: document.data.map(model => {
    if (!object(model) || model.object !== 'model') throw new ClientError('response');
    return { id: metadata(model.id, 128), object: 'model', owned_by: metadata(model.owned_by, 128) };
  }) };
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const configPath = path.resolve(options.config);
  let config;
  try { config = JSON.parse(await readFileBounded(configPath, 65536)); }
  catch { throw new ClientError('configuration'); }
  const host = config?.host ?? '127.0.0.1';
  const port = config?.port ?? 8790;
  if (!object(config) || config.version !== 1 || !['127.0.0.1', '::1'].includes(host) || !Number.isSafeInteger(port) || port < 1 || port > 65535 || typeof config.tokenFile !== 'string' || !config.tokenFile || config.tokenFile.length > 4096 || config.tokenFile.includes('\0')) {
    throw new ClientError('configuration');
  }
  const token = (await readFileBounded(path.resolve(path.dirname(configPath), config.tokenFile), 1024)).trim();
  if (!token || !/^[\x21-\x7e]+$/.test(token)) throw new ClientError('configuration');
  const requestLimit = Math.floor(limit(process.env.HARPY_LLM_MAX_REQUEST_BYTES, REQUEST_BYTES, MAX_BYTES));
  const responseLimit = Math.floor(limit(process.env.HARPY_LLM_MAX_RESPONSE_BYTES, RESPONSE_BYTES, MAX_BYTES));
  const timeout = Math.min(
    limit(config.requestTimeoutMs, 120000, 600000),
    limit(process.env.HARPY_LLM_TRANSPORT_TIMEOUT_MS, 120000, 600000),
    options.timeout ?? 120000,
  );
  const control = new AbortController();
  const timer = setTimeout(() => control.abort(), timeout);
  try {
    let body = null;
    if (options.mode === 'complete') {
      body = Buffer.from(JSON.stringify(await readInput(requestLimit, control.signal)), 'utf8');
      if (body.length > requestLimit) throw new ClientError('request-limit');
    }
    const route = options.mode === 'health' ? '/healthz' : options.mode === 'models' ? '/v1/models' : '/v1/chat/completions';
    const document = await request(host, port, token, route, body, responseLimit, control.signal);
    const result = options.mode === 'complete' ? normalize(document) : readiness(document, options.mode);
    const encoded = Buffer.from(JSON.stringify(result) + '\n', 'utf8');
    if (encoded.length > responseLimit) throw new ClientError('response-limit');
    process.stdout.write(encoded);
  } finally { clearTimeout(timer); }
}

main().catch(error => {
  const known = error instanceof ClientError;
  const code = known ? error.code : 'failed';
  const detail = known && code === 'http-status' && Number.isInteger(error.status) ? `HTTP ${error.status}` : code;
  process.stderr.write(`Local bridge client failed: ${detail}.\n`);
  process.exitCode = 1;
});
