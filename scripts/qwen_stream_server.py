import argparse
import time
import uuid

from flask import jsonify, request
from qwen_asr import Qwen3ASRModel
from qwen_asr.cli import demo_streaming


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--swap-space", type=float, default=0.0)
    parser.add_argument("--mm-processor-cache-gb", type=float, default=0.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--chunk-size-sec", type=float, default=0.8)
    parser.add_argument("--unfixed-chunk-num", type=int, default=4)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    demo_streaming.UNFIXED_CHUNK_NUM = args.unfixed_chunk_num
    demo_streaming.UNFIXED_TOKEN_NUM = args.unfixed_token_num
    demo_streaming.CHUNK_SIZE_SEC = args.chunk_size_sec
    demo_streaming.asr = Qwen3ASRModel.LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        swap_space=args.swap_space,
        mm_processor_cache_gb=args.mm_processor_cache_gb,
        enforce_eager=args.enforce_eager,
        max_new_tokens=32,
    )

    def api_start():
        payload = request.get_json(silent=True) or {}
        chunk_size = float(payload.get("chunk_size_sec", args.chunk_size_sec))
        chunk_size = max(0.4, min(2.0, chunk_size))
        unfixed_chunk_num = int(payload.get("unfixed_chunk_num", args.unfixed_chunk_num))
        unfixed_chunk_num = max(1, min(12, unfixed_chunk_num))
        unfixed_token_num = int(payload.get("unfixed_token_num", args.unfixed_token_num))
        unfixed_token_num = max(1, min(20, unfixed_token_num))
        session_id = uuid.uuid4().hex
        state = demo_streaming.asr.init_streaming_state(
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
            chunk_size_sec=chunk_size,
        )
        now = time.time()
        demo_streaming.SESSIONS[session_id] = demo_streaming.Session(
            state=state,
            created_at=now,
            last_seen=now,
        )
        return jsonify(
            {
                "session_id": session_id,
                "chunk_size_sec": chunk_size,
                "unfixed_chunk_num": unfixed_chunk_num,
                "unfixed_token_num": unfixed_token_num,
            }
        )

    demo_streaming.app.view_functions["api_start"] = api_start
    demo_streaming.app.run(
        host=args.host,
        port=args.port,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
