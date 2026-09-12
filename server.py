# Vocalizer WebSocket server.
#
# Spins up a local TLS (wss://) WebSocket server that renders Vocalizer JSON
# scenes to audio and streams the encoded result back to the client. Modelled
# after de-server/local-llama.py.
#
# Vocalizer only has a Python implementation, so there is no JS counterpart.
#
# Security note: this uses a simple shared-secret ("secret" file) mechanism and
# is intended for use with a trusted group. Put a real proxy in front for
# untrusted/public deployments.
#
# Per-connection behaviour:
#   * Each connection gets a private temporary directory. Uploaded audio files
#     land there and JSON scene references (file/ref) resolve relative to it.
#   * The directory is destroyed when the client disconnects.
#   * The directory is capped at MAX_DIR_BYTES (least-recently-used files are
#     evicted). Files unused for FILE_TTL_SECONDS are also cleaned up.
#
# Render jobs from all clients are serialized through a single FIFO queue so the
# (GPU-bound) model is only ever running one render at a time.

import asyncio
import gc
import hashlib
import io
import json as _json_mod
import os
import secrets
import shutil
import ssl
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, BrokenExecutor
from email.utils import formatdate
from http import HTTPStatus
from urllib.parse import urlparse, parse_qs

import numpy as np
import soundfile as sf
import websockets
from websockets.http11 import Response
from websockets.datastructures import Headers

from vocalizer import Vocalizer, VocalizerConfig, SoundLibrary, resolve_backend

# ── Configuration ─────────────────────────────────────────────────────────
PORT = int(os.getenv("PORT", "8222"))
HOST = os.getenv("HOST", "0.0.0.0")

DEV = os.getenv("DEV", "0") == "1"
ENABLE_UNLOAD = os.getenv("ENABLE_UNLOAD", "0") == "1"

MAX_UPLOAD_BYTES = 1 * 1024 * 1024        # 1 MB hard cap per uploaded file
MAX_DIR_BYTES = 100 * 1024 * 1024         # 100 MB per-connection quota
FILE_TTL_SECONDS = 30 * 60                # evict files unused for 30 minutes
CLEANUP_INTERVAL_SECONDS = 60             # how often the janitor task runs
STREAM_CHUNK_BYTES = 64 * 1024            # size of binary frames sent to client

# NOTE: OGG/Vorbis output is disabled. The bundled libsndfile (1.2.2) crashes
# natively when encoding OGG buffers longer than ~a few seconds, so all output
# is forced to MP3. Requests asking for "ogg" are transparently served as MP3.
SUPPORTED_OUTPUT_FORMATS = {"mp3": "MP3"}
DEFAULT_OUTPUT_FORMAT = "mp3"

BACKEND = resolve_backend()
SAMPLE_RATE = int(os.getenv("VOCALIZER_SAMPLE_RATE", "48000"))

SERVER_START_TIME = time.time()

# The shared Vocalizer instance (None while loading is deferred or unloaded).
VOCALIZER: Vocalizer = None  # set in main()

# Serializes model loading, unloading, and generation. Encoding doesn't use
# the model and happens outside this lock, allowing unload to release model
# memory as soon as generation finishes.
MODEL_LOCK: "asyncio.Lock" = None  # created inside main()

# FIFO render queue shared by every client.
RENDER_QUEUE: "asyncio.Queue" = None  # created inside main()

# Dedicated single-worker process pool used only for audio encoding. Encoding
# runs libsndfile (a native library) which can, on rare malformed buffers,
# crash at the C level. Isolating it in a child process means such a crash
# raises BrokenProcessPool here instead of taking the whole server down.
_ENCODE_POOL: "ProcessPoolExecutor" = None  # created lazily

# Registry of active connections, used by the cleanup janitor.
ACTIVE_CONNECTIONS = set()


# ── Static info page ──────────────────────────────────────────────────────
_INDEX_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
try:
    with open(_INDEX_HTML_PATH, "r", encoding="utf-8") as _f:
        INDEX_HTML_TEMPLATE = _f.read()
except Exception as _e:
    print(f"Warning: failed to load index.html template: {_e}")
    INDEX_HTML_TEMPLATE = "<html><body><h1>Vocalizer Server</h1><p>(template missing)</p></body></html>"


def _html_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _format_uptime(seconds: float) -> str:
    s = int(seconds)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, sec = divmod(s, 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    if h or d:
        parts.append(f"{h}h")
    if m or h or d:
        parts.append(f"{m}m")
    parts.append(f"{sec}s")
    return " ".join(parts)


def _mb(n: int) -> str:
    return f"{n / (1024 * 1024):.0f} MB"


def _render_index_html() -> str:
    model_id = VOCALIZER.model_id if VOCALIZER is not None else f"{BACKEND} (not loaded)"
    replacements = {
        "PROTOCOL": "wss",
        "PORT": str(PORT),
        "DEV_MODE": "DEV (insecure secret)" if DEV else "production",
        "SSL_MODE": "enabled",
        "MODEL_LOADED": "yes" if (VOCALIZER is not None and VOCALIZER.model is not None) else "no",
        "MODEL_ID": _html_escape(model_id),
        "SAMPLE_RATE": str(SAMPLE_RATE),
        "OUTPUT_FORMATS": ", ".join(SUPPORTED_OUTPUT_FORMATS.keys()),
        "MAX_UPLOAD": _mb(MAX_UPLOAD_BYTES),
        "MAX_DIR": _mb(MAX_DIR_BYTES),
        "ACTIVE_SESSIONS": str(len(ACTIVE_CONNECTIONS)),
        "QUEUE_LEN": str(RENDER_QUEUE.qsize() if RENDER_QUEUE is not None else 0),
        "UPTIME": _format_uptime(time.time() - SERVER_START_TIME),
    }
    out = INDEX_HTML_TEMPLATE
    for key, value in replacements.items():
        out = out.replace("{{" + key + "}}", str(value))
    return out


# ── Per-connection session (temp dir + upload bookkeeping) ─────────────────
class Session:
    """Holds the private temp directory and upload state for one connection."""

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="vocalizer_")
        # filename -> {"hash": str, "size": int, "last_used": float}
        self.files = {}
        # Set while awaiting a binary frame after an upload_audio_proceed.
        self.pending_upload = None  # {"filename": str, "hash": str}

    # -- filename safety -------------------------------------------------
    @staticmethod
    def _safe_name(filename: str) -> str:
        """Reject path traversal; only a plain basename is allowed."""
        if not filename or not isinstance(filename, str):
            raise ValueError("Invalid filename")
        base = os.path.basename(filename)
        if base != filename or base in ("", ".", ".."):
            raise ValueError("Invalid filename (no paths allowed)")
        return base

    def path_for(self, filename: str) -> str:
        return os.path.join(self.dir, self._safe_name(filename))

    # -- upload lifecycle ------------------------------------------------
    def has_matching(self, filename: str, sha256_hex: str) -> bool:
        name = self._safe_name(filename)
        info = self.files.get(name)
        if not info or info["hash"] != sha256_hex:
            return False
        # Confirm the file is still physically present.
        if not os.path.isfile(self.path_for(name)):
            self.files.pop(name, None)
            return False
        info["last_used"] = time.time()
        return True

    def store_upload(self, filename: str, data: bytes, expected_hash: str) -> str:
        name = self._safe_name(filename)
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(f"Upload exceeds {MAX_UPLOAD_BYTES} bytes limit")
        actual_hash = hashlib.sha256(data).hexdigest()
        if expected_hash and actual_hash != expected_hash:
            raise ValueError("Uploaded data hash does not match declared hash")
        path = self.path_for(name)
        with open(path, "wb") as f:
            f.write(data)
        self.files[name] = {"hash": actual_hash, "size": len(data), "last_used": time.time()}
        self._enforce_quota()
        return actual_hash

    def touch_all(self):
        """Mark every file as recently used (called at render time)."""
        now = time.time()
        for info in self.files.values():
            info["last_used"] = now

    def _total_bytes(self) -> int:
        return sum(info["size"] for info in self.files.values())

    def _enforce_quota(self):
        """Evict least-recently-used files until under MAX_DIR_BYTES."""
        while self._total_bytes() > MAX_DIR_BYTES and self.files:
            oldest = min(self.files.items(), key=lambda kv: kv[1]["last_used"])[0]
            self._remove(oldest)

    def cleanup_expired(self):
        """Delete files not used within FILE_TTL_SECONDS."""
        now = time.time()
        expired = [name for name, info in self.files.items()
                   if now - info["last_used"] > FILE_TTL_SECONDS]
        for name in expired:
            self._remove(name)

    def _remove(self, name: str):
        try:
            os.remove(self.path_for(name))
        except OSError:
            pass
        self.files.pop(name, None)

    def destroy(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        self.files.clear()


# ── Rendering ──────────────────────────────────────────────────────────────
def _sanitize_audio(audio: "np.ndarray", sample_rate: int) -> "np.ndarray":
    """Coerce a rendered audio buffer into something libsndfile can always
    encode safely: finite float32 samples clamped to [-1, 1].

    Malformed model output (NaN/Inf, or samples far outside [-1, 1]) is the
    usual trigger for a native Vorbis-encoder crash, so we scrub it here before
    the buffer ever reaches the encoder.
    """
    arr = np.asarray(audio)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    # Replace NaN -> 0 and +/-Inf -> +/-1.0.
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    # Clamp anything still out of range.
    np.clip(arr, -1.0, 1.0, out=arr)
    # Guarantee a contiguous 2D buffer.
    if arr.ndim == 1:
        arr = arr[:, None]
    return np.ascontiguousarray(arr)


def _encode_audio_worker(audio: "np.ndarray", sf_format: str, sample_rate: int) -> bytes:
    """Runs inside the encode subprocess. Writes to a real temp file (rather
    than BytesIO) so libsndfile's virtual-IO layer is never involved for
    seek-heavy formats like OGG Vorbis."""
    suffix = ".ogg" if sf_format == "OGG" else ".mp3"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.close(tmp_fd)
        sf.write(tmp_path, audio, sample_rate, format=sf_format)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _get_encode_pool() -> "ProcessPoolExecutor":
    """Return the shared encode pool, (re)creating it if missing or broken."""
    global _ENCODE_POOL
    if _ENCODE_POOL is None:
        _ENCODE_POOL = ProcessPoolExecutor(max_workers=1)
    return _ENCODE_POOL


def _reset_encode_pool():
    """Tear down a broken encode pool so the next render gets a fresh one."""
    global _ENCODE_POOL
    pool = _ENCODE_POOL
    _ENCODE_POOL = None
    if pool is not None:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


async def _encode_audio(audio: "np.ndarray", output_format: str) -> bytes:
    """Sanitize and encode a rendered audio array in an isolated subprocess.

    A native crash in the encoder surfaces here as BrokenExecutor/OSError, which
    the render worker turns into a normal error response — the main server
    process keeps running.
    """
    sf_format = SUPPORTED_OUTPUT_FORMATS[output_format]
    safe = _sanitize_audio(audio, SAMPLE_RATE)

    loop = asyncio.get_running_loop()
    pool = _get_encode_pool()
    try:
        return await loop.run_in_executor(pool, _encode_audio_worker, safe, sf_format, SAMPLE_RATE)
    except (BrokenExecutor, OSError) as e:
        # The encode subprocess died (likely a native crash). Recycle the pool
        # and report a clean failure instead of crashing the server.
        _reset_encode_pool()
        raise RuntimeError(f"Audio encoding failed (encoder subprocess crashed): {e}") from e


def _render_blocking(vocalizer: Vocalizer, payload: dict, session_dir: str) -> np.ndarray:
    """Runs on an executor thread. Points the shared Vocalizer's library at the
    connection's temp dir so file/ref references resolve to uploaded files."""
    vocalizer.library = SoundLibrary(session_dir)
    vocalizer.config.sound_library_dir = session_dir
    return vocalizer.render_json(payload)


def _create_vocalizer() -> Vocalizer:
    """Construct and load a fresh Vocalizer model."""
    return Vocalizer(VocalizerConfig(
        output_sample_rate=SAMPLE_RATE,
    ))


def _release_vocalizer(vocalizer: Vocalizer):
    """Release model references and return cached accelerator memory."""
    # The awaiting coroutine retains the lightweight Vocalizer wrapper until
    # this function returns, so detach the heavyweight model explicitly before
    # collecting objects and clearing the allocator cache.
    vocalizer.model = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # CPU-only installations and partially initialized CUDA runtimes don't
        # need any additional cleanup.
        pass


async def _render_worker():
    """Single consumer of RENDER_QUEUE. Guarantees FIFO, one render at a time.

    Wrapped in an outer restart loop so that any unhandled BaseException (e.g.
    an unexpected native crash propagated via run_in_executor) is printed and
    the worker restarts immediately rather than silently dying.
    """
    loop = asyncio.get_running_loop()
    while True:
        job = None
        try:
            job = await RENDER_QUEUE.get()
        except asyncio.CancelledError:
            print("[render_worker] cancelled — stopping.")
            return

        websocket = job["websocket"]
        rid = job["rid"]
        payload = job["payload"]
        output_format = job["output_format"]
        session = job["session"]
        future = job["future"]
        try:
            session.touch_all()
            print(f"[render] {rid}: start (format={output_format}) from {websocket.remote_address}")
            print(f"[render] {rid}: payload={payload}")

            print(f"[render] {rid}: generating audio…")
            async with MODEL_LOCK:
                vocalizer = VOCALIZER
                if vocalizer is None:
                    raise RuntimeError(
                        "Vocalizer model is unloaded; call load_model before rendering"
                    )
                audio = await loop.run_in_executor(
                    None, _render_blocking, vocalizer, payload, session.dir,
                )
            print(f"[render] {rid}: generation done, shape={audio.shape} — encoding to {output_format}…")

            data = await _encode_audio(audio, output_format)
            print(f"[render] {rid}: encoded {len(data):,} bytes — streaming to client…")

            await websocket.send(_json({"type": "render_start", "rid": rid, "format": output_format}))
            for i in range(0, len(data), STREAM_CHUNK_BYTES):
                await websocket.send(data[i:i + STREAM_CHUNK_BYTES])
            await websocket.send(_json({"type": "render_done", "rid": rid, "bytes": len(data)}))
            print(f"[render] {rid}: done.")
            if not future.done():
                future.set_result(True)
        except Exception as e:
            print(f"[render] {rid}: ERROR — {e}")
            traceback.print_exc()
            try:
                await websocket.send(_json({"type": "error", "rid": rid, "message": str(e)}))
            except Exception:
                pass
            if not future.done():
                future.set_result(False)
        except BaseException as e:
            # CancelledError, KeyboardInterrupt, or a native crash re-raised from
            # run_in_executor.  Log it, unblock the caller, then re-raise.
            # (finally: RENDER_QUEUE.task_done() still runs after the raise.)
            print(f"[render] {rid}: FATAL {type(e).__name__}: {e}")
            traceback.print_exc()
            if not future.done():
                future.set_result(False)
            raise
        finally:
            RENDER_QUEUE.task_done()


# ── WebSocket handling ─────────────────────────────────────────────────────
def _json(obj) -> str:
    return _json_mod.dumps(obj)


async def handle_client(websocket):
    print("Client connected")
    session = Session()
    ACTIVE_CONNECTIONS.add(session)

    await websocket.send(_json({
        "type": "ready",
        "message": "Vocalizer is ready",
        "supported_output_formats": list(SUPPORTED_OUTPUT_FORMATS.keys()),
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_session_bytes": MAX_DIR_BYTES,
        "file_ttl_seconds": FILE_TTL_SECONDS,
        "sample_rate": SAMPLE_RATE,
        "supports_parallel_requests": False,
        "supports_model_unload": ENABLE_UNLOAD,
        "model_loaded": VOCALIZER is not None,
    }))

    try:
        async for message in websocket:
            # ── Binary frame: the payload of a pending upload ──────────
            if isinstance(message, (bytes, bytearray)):
                await _handle_binary(websocket, session, bytes(message))
                continue

            # ── Text frame: a JSON control message ─────────────────────
            rid = "no-rid"
            try:
                data = _json_mod.loads(message)
                rid = data.get("rid", "no-rid")
                action = data.get("action")

                if action == "upload_audio":
                    await _handle_upload_request(websocket, session, data, rid)
                elif action == "render_json":
                    await _handle_render_request(websocket, session, data, rid)
                elif action == "load_model":
                    await _handle_load_model(websocket, rid)
                elif action == "unload_model":
                    await _handle_unload_model(websocket, rid)
                elif action == "ping":
                    await websocket.send(_json({"type": "pong", "rid": rid}))
                else:
                    await websocket.send(_json({
                        "type": "error", "rid": rid,
                        "message": f"Unknown action: {action}",
                    }))
            except Exception as e:
                print(f"Error handling message: {e}")
                await websocket.send(_json({"type": "error", "rid": rid, "message": str(e)}))
    except websockets.ConnectionClosedOK:
        print("Client disconnected normally")
    except websockets.ConnectionClosedError as e:
        print(f"Client disconnected abnormally: code={e.code} reason={e.reason}")
    finally:
        ACTIVE_CONNECTIONS.discard(session)
        session.destroy()
        print("Session cleaned up")


async def _handle_upload_request(websocket, session: Session, data: dict, rid: str):
    filename = data.get("filename")
    sha256_hex = data.get("hash")
    if not filename or not sha256_hex:
        raise ValueError("upload_audio requires 'filename' and 'hash'")

    # Validate the name early (raises on traversal attempts).
    session._safe_name(filename)

    if session.has_matching(filename, sha256_hex):
        # Already present with the same hash; the client can skip sending it.
        session.pending_upload = None
        await websocket.send(_json({
            "type": "upload_audio_skip", "rid": rid, "filename": filename,
        }))
    else:
        # Await one binary frame carrying the file bytes.
        session.pending_upload = {"filename": filename, "hash": sha256_hex, "rid": rid}
        await websocket.send(_json({
            "type": "upload_audio_proceed", "rid": rid, "filename": filename,
        }))


async def _handle_binary(websocket, session: Session, data: bytes):
    pending = session.pending_upload
    if not pending:
        await websocket.send(_json({
            "type": "error", "rid": "no-rid",
            "message": "Unexpected binary frame (no upload in progress)",
        }))
        return
    session.pending_upload = None
    rid = pending.get("rid", "no-rid")
    try:
        actual_hash = session.store_upload(pending["filename"], data, pending["hash"])
        await websocket.send(_json({
            "type": "upload_audio_done", "rid": rid,
            "filename": pending["filename"], "hash": actual_hash,
            "size": len(data),
        }))
    except Exception as e:
        await websocket.send(_json({"type": "error", "rid": rid, "message": str(e)}))


async def _handle_render_request(websocket, session: Session, data: dict, rid: str):
    if VOCALIZER is None:
        raise RuntimeError(
            "Vocalizer model is unloaded; call load_model before rendering"
        )

    payload = data.get("payload")
    if not payload:
        raise ValueError("render_json requires a 'payload'")
    output_format = (data.get("output_format") or DEFAULT_OUTPUT_FORMAT).lower()
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        # OGG (and anything else) is not supported by this build; fall back to
        # MP3 rather than failing the request.
        print(f"[render] {rid}: output_format '{output_format}' unsupported — using {DEFAULT_OUTPUT_FORMAT}")
        output_format = DEFAULT_OUTPUT_FORMAT

    future = asyncio.get_running_loop().create_future()
    position = RENDER_QUEUE.qsize()
    await RENDER_QUEUE.put({
        "websocket": websocket,
        "rid": rid,
        "payload": payload,
        "output_format": output_format,
        "session": session,
        "future": future,
    })
    await websocket.send(_json({"type": "queued", "rid": rid, "position": position}))
    # Wait for this job to finish so we don't interleave two renders on one socket.
    await future


def _require_unload_enabled(action: str):
    if not ENABLE_UNLOAD:
        raise RuntimeError(
            f"{action} is disabled; start the server with ENABLE_UNLOAD=1"
        )


async def _handle_load_model(websocket, rid: str):
    """Load the shared model, treating an already-loaded model as success."""
    global VOCALIZER
    _require_unload_enabled("load_model")

    async with MODEL_LOCK:
        if VOCALIZER is None:
            print(f"[model] {rid}: loading {BACKEND}...")
            loop = asyncio.get_running_loop()
            VOCALIZER = await loop.run_in_executor(None, _create_vocalizer)
            print(f"[model] {rid}: loaded.")

    await websocket.send(_json({"type": "model_loaded", "rid": rid}))


async def _handle_unload_model(websocket, rid: str):
    """Unload the shared model after any active generation has completed."""
    global VOCALIZER
    _require_unload_enabled("unload_model")

    async with MODEL_LOCK:
        vocalizer = VOCALIZER
        VOCALIZER = None
        if vocalizer is not None:
            print(f"[model] {rid}: unloading...")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _release_vocalizer, vocalizer)
            print(f"[model] {rid}: unloaded.")

    await websocket.send(_json({"type": "model_unloaded", "rid": rid}))


# ── Background janitor ─────────────────────────────────────────────────────
async def _cleanup_task():
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        for session in list(ACTIVE_CONNECTIONS):
            try:
                session.cleanup_expired()
            except Exception as e:
                print(f"Cleanup error: {e}")


# ── HTTP handshake / info page / auth ──────────────────────────────────────
async def process_request(connection, request):
    parsed = urlparse(request.path)

    is_websocket = request.headers.get("Upgrade", "").lower() == "websocket"

    # Serve the info page only for ordinary browser (non-WebSocket) requests to
    # the root. WebSocket upgrades on any path fall through to auth + handshake.
    if not is_websocket and parsed.path in ("/", "/index.html"):
        body = _render_index_html().encode("utf-8")
        headers = Headers([
            ("Date", formatdate(usegmt=True)),
            ("Connection", "close"),
            ("Content-Type", "text/html; charset=utf-8"),
            ("Cache-Control", "no-store"),
            ("Content-Length", str(len(body))),
        ])
        return Response(HTTPStatus.OK.value, HTTPStatus.OK.phrase, headers, body)

    # Any plain HTTP request on an unknown path (e.g. /favicon.ico) gets a
    # silent 404. Only WebSocket upgrade requests proceed to auth.
    if not is_websocket:
        return connection.respond(HTTPStatus.NOT_FOUND, "Not found")

    query_params = parse_qs(parsed.query)
    secret = query_params.get("secret", [None])[0]

    if not DEV:
        try:
            with open("./secret", "r") as f:
                expected_secret = f.read().strip()
        except Exception:
            expected_secret = secrets.token_hex(64)
            with open("./secret", "w") as f:
                f.write(expected_secret)
            print(f"Generated new Secret key and saved to ./secret: {expected_secret}")
    else:
        expected_secret = "dev-secret-12345678900abcdef"

    if secret != expected_secret:
        print("Unauthorized connection attempt with invalid secret")
        return connection.respond(HTTPStatus.UNAUTHORIZED, "Unauthorized")

    return None  # continue the WebSocket handshake


# ── Startup ────────────────────────────────────────────────────────────────
async def main():
    global MODEL_LOCK, RENDER_QUEUE, VOCALIZER
    MODEL_LOCK = asyncio.Lock()
    RENDER_QUEUE = asyncio.Queue()

    if ENABLE_UNLOAD:
        print("Model loading deferred because ENABLE_UNLOAD=1.")
    else:
        print(f"Loading {BACKEND} model...")
        VOCALIZER = _create_vocalizer()
        print("Model loaded.")

    ssl_context = None
    try:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
        print("SSL context created with cert.pem and key.pem")
    except Exception as e:
        print(f"Failed to create SSL context: {e}")
        print("Run ./create-ssl-keys.sh to generate cert.pem and key.pem first.")
        sys.exit(1)

    asyncio.create_task(_render_worker())
    asyncio.create_task(_cleanup_task())

    # Pre-warm the isolated encode pool so the (Windows spawn) startup cost is
    # paid now rather than on the first client render.
    try:
        loop = asyncio.get_running_loop()
        pool = _get_encode_pool()
        warm = _sanitize_audio(np.zeros((1, 2), dtype=np.float32), SAMPLE_RATE)
        await loop.run_in_executor(pool, _encode_audio_worker, warm, "MP3", SAMPLE_RATE)
        print("Encode subprocess pool ready.")
    except Exception as e:
        print(f"Warning: failed to pre-warm encode pool: {e}")

    server = await websockets.serve(
        handle_client, HOST, PORT,
        process_request=process_request,
        ssl=ssl_context,
        max_size=MAX_UPLOAD_BYTES + 4096,  # cap incoming frame size (uploads + small headroom)
    )
    print(f"Vocalizer server listening on wss://{HOST}:{PORT}/")
    await server.serve_forever()


if __name__ == "__main__":
    if not DEV:
        try:
            with open("./secret", "r"):
                print("Using Secret key from ./secret for authentication.")
        except Exception:
            print("No existing secret key found. A new one will be generated and saved to ./secret.")
            new_secret = secrets.token_hex(64)
            with open("./secret", "w") as f:
                f.write(new_secret)
            print(f"Generated new Secret key and saved to ./secret: {new_secret}")

    print("DEV mode:", DEV)
    print("ENABLE_UNLOAD:", ENABLE_UNLOAD)
    asyncio.run(main())
