import argparse

from qwen_asr import Qwen3ASRModel
from qwen_asr.cli import demo_streaming


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--max-model-len", type=int, default=16384)
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
        max_new_tokens=32,
    )
    demo_streaming.app.run(
        host=args.host,
        port=args.port,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
