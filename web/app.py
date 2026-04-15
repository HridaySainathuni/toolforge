from __future__ import annotations

import json
import queue
import threading
import uuid

from flask import Flask, Response, jsonify, render_template, request

import anthropic as _anthropic

from agent.librarian import Librarian
from agent.loop import AgentLoop
from config import Config
from library.tool_library import ToolLibrary

app = Flask(__name__, template_folder="templates")

# Will be set by main.py
tool_library: ToolLibrary | None = None
task_queues: dict[str, queue.Queue] = {}


def init_app(library: ToolLibrary) -> Flask:
    global tool_library
    tool_library = library
    return app


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/task", methods=["POST"])
def start_task():
    data = request.get_json()
    task_text = data.get("task", "").strip()
    if not task_text:
        return jsonify({"error": "No task provided"}), 400

    task_id = str(uuid.uuid4())
    q: queue.Queue = queue.Queue()
    task_queues[task_id] = q

    def run_agent():
        try:
            loop = AgentLoop(tool_library=tool_library, event_queue=q)
            loop.run(task_text)
        except Exception as e:
            q.put({"type": "error", "content": str(e)})
        finally:
            q.put({"type": "done"})

    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()

    return jsonify({"task_id": task_id})


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
