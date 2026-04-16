from __future__ import annotations

import json
import queue
import threading
import uuid

from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os

import anthropic as _anthropic

from agent.librarian import Librarian
from agent.loop import AgentLoop
from config import Config
from library.tool_library import ToolLibrary

app = Flask(__name__, template_folder="templates", static_folder="static")

# Will be set by main.py
tool_library: ToolLibrary | None = None
task_queues: dict[str, queue.Queue] = {}
task_stop_events: dict[str, threading.Event] = {}


def init_app(library: ToolLibrary) -> Flask:
    global tool_library
    tool_library = library
    return app


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(os.path.join(static_dir, "favicon.ico")):
        return send_from_directory(static_dir, "favicon.ico")
    return "", 204


@app.route("/api/task", methods=["POST"])
def start_task():
    data = request.get_json()
    task_text = data.get("task", "").strip()
    if not task_text:
        return jsonify({"error": "No task provided"}), 400

    task_id = str(uuid.uuid4())
    q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    task_queues[task_id] = q
    task_stop_events[task_id] = stop_event

    def run_agent():
        try:
            loop = AgentLoop(tool_library=tool_library, event_queue=q, stop_event=stop_event)
            loop.run(task_text)
        except Exception as e:
            q.put({"type": "error", "content": str(e)})
        finally:
            q.put({"type": "done"})

    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()

    return jsonify({"task_id": task_id})


@app.route("/api/task/<task_id>/stop", methods=["POST"])
def stop_task(task_id: str):
    stop_event = task_stop_events.get(task_id)
    if stop_event:
        stop_event.set()
    q = task_queues.get(task_id)
    if q:
        q.put({"type": "done"})
    return jsonify({"ok": True})


@app.route("/api/task/<task_id>/stream")
def stream_task(task_id: str):
    q = task_queues.get(task_id)
    if q is None:
        return jsonify({"error": "Task not found"}), 404

    def generate():
        while True:
            try:
                event = q.get(timeout=120)
                yield f"data: {json.dumps(event, default=str)}\n\n"
                if event.get("type") in ("final_answer", "impossible", "error", "done"):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

        # Cleanup
        task_queues.pop(task_id, None)
        task_stop_events.pop(task_id, None)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/tools")
def list_tools():
    if tool_library is None:
        return jsonify([])
    return jsonify(tool_library.get_all_tools_public())


@app.route("/api/tools/<name>/source")
def get_tool_source(name: str):
    if tool_library is None:
        return jsonify({"error": "Library not initialized"}), 500
    source = tool_library.get_source_code(name)
    if source is None:
        return jsonify({"error": "Tool not found"}), 404
    return jsonify({"name": name, "source": source})


@app.route("/api/tools/<name>", methods=["DELETE"])
def delete_tool(name: str):
    if tool_library is None:
        return jsonify({"error": "Library not initialized"}), 500
    tool_library.delete_tool(name)
    return jsonify({"ok": True})


@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    filename = secure_filename(f.filename)
    dest = os.path.join(Config.WORKSPACE_DIR, filename)
    f.save(dest)
    return jsonify({"filename": filename, "path": dest, "size": os.path.getsize(dest)})


@app.route("/api/workspace", methods=["GET"])
def get_workspace():
    return jsonify({"workspace": Config.WORKSPACE_DIR})


@app.route("/api/workspace", methods=["POST"])
def set_workspace():
    data = request.get_json()
    path = data.get("path", "").strip()
    if not path or not os.path.isdir(path):
        return jsonify({"error": f"Not a valid directory: {path}"}), 400
    Config.WORKSPACE_DIR = os.path.abspath(path)
    return jsonify({"workspace": Config.WORKSPACE_DIR})


@app.route("/api/librarian/run", methods=["POST"])
def run_librarian():
    if tool_library is None:
        return jsonify({"error": "Library not initialized"}), 500
    client = _anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    librarian = Librarian(library=tool_library, client=client)
    report = librarian.run_pass()
    return jsonify({
        "tools_merged": report.tools_merged,
        "tools_refactored": report.tools_refactored,
        "library_size_before": report.library_size_before,
        "library_size_after": report.library_size_after,
        "details": report.details,
    })
