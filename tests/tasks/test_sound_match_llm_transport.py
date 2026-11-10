"""Local fake servers/processes verify bounds without harness login or generation."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "sound_match_llm_transport", _ROOT / "research/sound_matching/llm/transport.py"
)
transport = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(transport)
MESSAGES = [
    {"role": "system", "content": "Use plain text."},
    {"role": "user", "content": "Fixture"},
]
SECRET = "fixture-only-client-token-never-real"


def completion(content="fixture response"):
    return {
        "id": "chatcmpl-fixture",
        "model": "fixture-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }


@contextmanager
def local_server(**options):
    calls = []
    raw = options.get("raw", json.dumps(completion()).encode())

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            calls.append(
                {
                    "path": self.path,
                    "headers": dict(self.headers),
                    "body": self.rfile.read(int(self.headers.get("Content-Length", "0"))),
                }
            )
            time.sleep(options.get("delay", 0))
            try:
                self.send_response(options.get("status", 200))
                self.send_header("Content-Type", "application/json")
                self.send_header("Connection", "close")
                if not options.get("omit_length", False):
                    self.send_header("Content-Length", str(len(raw)))
                for name, value in options.get("headers", {}).items():
                    self.send_header(name, value)
                self.end_headers()
                if options.get("trickle", False):
                    for byte in raw:
                        self.wfile.write(bytes((byte,)))
                        self.wfile.flush()
                        time.sleep(0.02)
                else:
                    self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


@pytest.fixture
def token_file(tmp_path):
    filename = tmp_path / "client.token"
    filename.write_text(SECRET + "\n")
    return filename


def test_http_sends_only_supported_request_and_normalizes_result(token_file, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    with local_server() as (url, calls):
        result = transport.HttpTransport(url, token_file=token_file).complete(
            model="fixture-model", messages=MESSAGES
        )
    assert result == {
        "content": "fixture response",
        "usage": completion()["usage"],
        "response_id": "chatcmpl-fixture",
        "response_model": "fixture-model",
    }
    assert len(calls) == 1
    assert calls[0]["path"] == "/v1/chat/completions"
    assert json.loads(calls[0]["body"]) == {
        "model": "fixture-model",
        "messages": MESSAGES,
        "stream": False,
    }
    assert calls[0]["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert "Origin" not in calls[0]["headers"]
    assert calls[0]["headers"]["Host"] == url.removeprefix("http://").removesuffix("/v1")


def test_http_environment_token_is_explicit_and_not_persisted(monkeypatch):
    monkeypatch.setenv("HARPY_TEST_LOCAL_TOKEN", SECRET)
    with local_server() as (url, calls):
        result = transport.HttpTransport(url, token_env="HARPY_TEST_LOCAL_TOKEN").complete(
            model="fixture-model", messages=MESSAGES
        )
    assert result["content"] == "fixture response" and len(calls) == 1


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:8790/v1",
        "http://localhost:8790/v1",
        "http://127.0.0.2:8790/v1",
        "http://192.168.1.1/v1",
        "http://example.com/v1",
        "file:///v1",
        "http://user:secret@127.0.0.1/v1",
        "http://127.0.0.1/v1?x=secret",
        "http://127.0.0.1/v1#secret",
        "http://127.0.0.1/v1?",
        "http://127.0.0.1/other",
        "http://127.0.0.1:0/v1",
        "http://127.0.0.1:99999/v1",
        "http://127.0.0.1/\r\n",
    ],
)
def test_http_rejects_non_loopback_and_ambiguous_urls(url, token_file):
    with pytest.raises(transport.TransportError, match="loopback") as error:
        transport.HttpTransport(url, token_file=token_file)
    assert "secret" not in str(error.value)


def test_ipv6_literal_can_be_configured_without_connecting(token_file):
    configured = transport.HttpTransport("http://[::1]:8790/v1/", token_file=token_file)
    assert (configured.host, configured.port) == ("::1", 8790)


def test_redirect_is_not_followed_or_retried(token_file):
    with local_server(status=302, headers={"Location": "http://127.0.0.1:1/stolen"}) as (
        url,
        calls,
    ):
        with pytest.raises(transport.TransportError, match="status 302"):
            transport.HttpTransport(url, token_file=token_file).complete(
                model="fixture", messages=MESSAGES
            )
        assert len(calls) == 1


@pytest.mark.parametrize("omit_length", [False, True])
def test_http_response_bound_with_and_without_content_length(token_file, omit_length):
    with local_server(raw=b"x" * 1000, omit_length=omit_length) as (url, calls):
        with pytest.raises(transport.TransportError, match="byte limit"):
            transport.HttpTransport(url, token_file=token_file, max_response_bytes=200).complete(
                model="fixture", messages=MESSAGES
            )
        assert len(calls) == 1


def test_http_request_bound_prevents_any_attempt(token_file):
    with local_server() as (url, calls):
        with pytest.raises(transport.TransportError, match="request exceeds"):
            transport.HttpTransport(url, token_file=token_file, max_request_bytes=10).complete(
                model="fixture", messages=MESSAGES
            )
        assert not calls


@pytest.mark.parametrize("options", [{"delay": 0.5}, {"trickle": True}])
def test_http_wall_timeout_is_one_attempt_even_with_trickle(token_file, options):
    with local_server(**options) as (url, calls):
        started = time.monotonic()
        with pytest.raises(transport.TransportError, match="timed out"):
            transport.HttpTransport(url, token_file=token_file, timeout_seconds=0.1).complete(
                model="fixture", messages=MESSAGES
            )
        assert time.monotonic() - started < 1.5
        assert len(calls) == 1


@pytest.mark.parametrize(
    "raw",
    [
        SECRET.encode(),
        b'{"choices": [], "choices": []}',
        json.dumps({**completion(), "usage": {"total_tokens": -1}}).encode(),
        json.dumps(
            {
                **completion(),
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "x", "tool_calls": []},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode(),
        json.dumps(
            {
                **completion(),
                "choices": [
                    {"message": {"role": "assistant", "content": "x"}, "finish_reason": "length"}
                ],
            }
        ).encode(),
    ],
)
def test_http_invalid_response_never_leaks_body(token_file, raw):
    with (
        local_server(raw=raw) as (url, _calls),
        pytest.raises(transport.TransportError) as error,
    ):
        transport.HttpTransport(url, token_file=token_file).complete(
            model="fixture", messages=MESSAGES
        )
    assert SECRET not in str(error.value)


def test_http_error_status_never_exposes_upstream_message(token_file):
    with local_server(status=401, raw=SECRET.encode()) as (url, calls):
        with pytest.raises(transport.TransportError, match="status 401") as error:
            transport.HttpTransport(url, token_file=token_file).complete(
                model="fixture", messages=MESSAGES
            )
        assert len(calls) == 1
    assert SECRET not in str(error.value)


def test_bad_authentication_is_rejected_before_network(token_file):
    token_file.write_text("secret\r\ninjected: value")
    with local_server() as (url, calls):
        with pytest.raises(transport.TransportError, match="authentication") as error:
            transport.HttpTransport(url, token_file=token_file).complete(
                model="fixture", messages=MESSAGES
            )
        assert not calls
    assert "injected" not in str(error.value)


def command_fixture(tmp_path, source):
    script = tmp_path / "completion_fixture.py"
    script.write_text("import json, os, sys, time\nfrom pathlib import Path\n" + source)
    return (sys.executable, str(script))


def test_command_json_stdin_and_normalized_stdout(tmp_path):
    command = command_fixture(
        tmp_path,
        """
request=json.load(sys.stdin)
assert set(request)=={'model','messages','stream'} and request['stream'] is False
assert request['model']=='fixture'
print(json.dumps({'content':request['messages'][-1]['content'], 'usage':None,
                  'response_id':'fixture-id','response_model':request['model']}))
""",
    )
    result = transport.CommandTransport(command).complete(model="fixture", messages=MESSAGES)
    assert result == {
        "content": "Fixture",
        "usage": None,
        "response_id": "fixture-id",
        "response_model": "fixture",
    }


@pytest.mark.parametrize(
    "source",
    [
        f"sys.stderr.write({SECRET!r}); sys.exit(3)",
        f"print({SECRET!r})",
        "print(json.dumps({'content':'x','usage':{'total_tokens':True}}))",
    ],
)
def test_command_failures_are_safe(tmp_path, source):
    with pytest.raises(transport.TransportError) as error:
        transport.CommandTransport(command_fixture(tmp_path, source)).complete(
            model="fixture", messages=MESSAGES
        )
    assert SECRET not in str(error.value)


def test_command_timeout_stops_owned_process_and_does_not_retry(tmp_path):
    calls = tmp_path / "calls"
    pid = tmp_path / "pid"
    command = command_fixture(
        tmp_path,
        f"""
json.load(sys.stdin)
Path({str(calls)!r}).write_text('one attempt')
Path({str(pid)!r}).write_text(str(os.getpid()))
time.sleep(30)
""",
    )
    started = time.monotonic()
    with pytest.raises(transport.TransportError, match="timed out"):
        transport.CommandTransport(command, timeout_seconds=0.3).complete(
            model="fixture", messages=MESSAGES
        )
    assert time.monotonic() - started < 3
    assert calls.read_text() == "one attempt"
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid.read_text()), 0)


def test_command_output_limit_stops_process_early(tmp_path):
    command = command_fixture(
        tmp_path, "sys.stdout.write('x'*100000); sys.stdout.flush(); time.sleep(30)"
    )
    started = time.monotonic()
    with pytest.raises(transport.TransportError, match="byte limit"):
        transport.CommandTransport(command, max_response_bytes=200, timeout_seconds=5).complete(
            model="fixture", messages=MESSAGES
        )
    assert time.monotonic() - started < 3


@pytest.mark.parametrize(
    "command",
    [
        ("sh", "-c", "ignored"),
        ("/bin/bash", "-c", "ignored"),
        ("C:\\Windows\\System32\\cmd.exe", "/c", "ignored"),
        ("powershell.exe", "-Command", "ignored"),
        ("pwsh", "-c", "ignored"),
        ("wrapper.cmd",),
        ("wrapper.ps1",),
        ("native", "--token=secret"),
        ("native", "--api-key", "secret"),
    ],
)
def test_command_rejects_shells_and_direct_secret_arguments(command):
    with pytest.raises(transport.TransportError, match="unsupported") as error:
        transport.CommandTransport(command)
    assert "secret" not in str(error.value).replace("secret arguments", "")


def test_command_request_limit_prevents_launch(tmp_path):
    marker = tmp_path / "started"
    command = command_fixture(tmp_path, f"Path({str(marker)!r}).write_text('started')")
    with pytest.raises(transport.TransportError, match="request exceeds"):
        transport.CommandTransport(command, max_request_bytes=10).complete(
            model="fixture", messages=MESSAGES
        )
    assert not marker.exists()


# This fixture also runs directly with native Windows Node: its server, client,
# temporary config and synthetic token all stay inside that same OS namespace.
NODE_FIXTURE = r"""
import assert from 'node:assert/strict';
import http from 'node:http';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {spawn} from 'node:child_process';
const [client,scenario]=process.argv.slice(1);
const root=await fs.mkdtemp(path.join(os.tmpdir(),'harpy-bridge-fixture-'));
const token='fake-local-token-not-a-real-credential';
const timers=[];
let calls=0;
const server=http.createServer(async(req,res)=>{
  calls++;
  assert.equal(req.headers.authorization,'Bearer '+token);
  assert.equal(req.headers.origin,undefined);
  assert.equal(req.headers.host,'127.0.0.1:'+server.address().port);
  let input='';for await(const chunk of req)input+=chunk;
  if(scenario==='health'||scenario==='models'){
    assert.equal(req.method,'GET');
    assert.equal(req.url,scenario==='health'?'/healthz':'/v1/models');
    res.end(JSON.stringify(scenario==='health'?{status:'ok',version:'fixture',active_requests:0}:{object:'list',data:[{id:'fixture',object:'model',owned_by:'fixture'}]}));return;
  }
  assert.equal(req.url,'/v1/chat/completions');
  const request=JSON.parse(input);
  assert.deepEqual(Object.keys(request).sort(),['messages','model','stream']);
  assert.equal(request.model,'fixture');assert.equal(request.stream,false);
  if(scenario==='redirect'){res.writeHead(302,{Location:'http://127.0.0.1:1/no-redirect'});res.end(token);return;}
  if(scenario==='oversize'){res.end('x'.repeat(1000));return;}
  if(scenario==='invalid'){res.end(token);return;}
  if(scenario==='timeout'){timers.push(setTimeout(()=>res.end('{}'),2000));return;}
  res.end(JSON.stringify({id:'fixture-id',model:'fixture',choices:[{message:{role:'assistant',content:'fixture-result'},finish_reason:'stop'}],usage:{prompt_tokens:1,completion_tokens:2,total_tokens:3}}));
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
try{
  const tokenPath=path.join(root,'client.token');await fs.writeFile(tokenPath,token+'\n');
  const config=path.join(root,'bridge.json');
  await fs.writeFile(config,JSON.stringify({
    version:1,host:scenario==='nonlocal'?'192.0.2.1':'127.0.0.1',
    port:server.address().port,tokenFile:tokenPath,requestTimeoutMs:1000
  }));
  const args=[client,'--config',config,'--timeout-seconds',scenario==='timeout'?'0.15':'2'];
  if(scenario==='health'||scenario==='models')args.push('--'+scenario);
  const child=spawn(process.execPath,args,{
    shell:false,windowsHide:true,stdio:['pipe','pipe','pipe'],
    env:{...process.env,HARPY_LLM_MAX_RESPONSE_BYTES:scenario==='oversize'?'200':'1048576'}
  });
  let stdout='',stderr='';
  child.stdout.on('data',chunk=>stdout+=chunk);child.stderr.on('data',chunk=>stderr+=chunk);
  child.stdin.end(JSON.stringify({
    model:'fixture',messages:[{role:'user',content:'fixture prompt'}],stream:false
  }));
  const code=await new Promise((resolve,reject)=>{
    child.once('error',reject);child.once('exit',resolve)
  });
  assert.ok(!stdout.includes(token)&&!stderr.includes(token));
  if(['success','health','models'].includes(scenario)){
    assert.equal(code,0,stderr);const result=JSON.parse(stdout);
    if(scenario==='success')assert.deepEqual(result,{content:'fixture-result',usage:{prompt_tokens:1,completion_tokens:2,total_tokens:3},response_id:'fixture-id',response_model:'fixture'});
    if(scenario==='health')assert.equal(result.status,'ok');
    if(scenario==='models')assert.equal(result.data[0].id,'fixture');
  }else{assert.notEqual(code,0);assert.equal(stdout,'');}
  assert.equal(calls,scenario==='nonlocal'?0:1);
  console.log(JSON.stringify({scenario,calls,passed:true}));
}finally{
  timers.forEach(clearTimeout);server.closeAllConnections();
  await new Promise(resolve=>server.close(resolve));
  const safeRoot=path.resolve(root);
  assert.equal(path.dirname(safeRoot),path.resolve(os.tmpdir()));
  assert.ok(path.basename(safeRoot).startsWith('harpy-bridge-fixture-'));
  await fs.rm(safeRoot,{recursive:true,force:true});
}
"""


@pytest.mark.parametrize(
    "scenario",
    ["success", "redirect", "oversize", "timeout", "invalid", "nonlocal", "health", "models"],
)
def test_node_bridge_client_uses_only_local_fake_service(scenario):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Native Node executable is not available in this test OS namespace")
    client = _ROOT / "research/sound_matching/llm/bridge_client.mjs"
    result = subprocess.run(
        [node, "--input-type=module", "-e", NODE_FIXTURE, str(client), scenario],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["passed"] is True
