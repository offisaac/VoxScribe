import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from voxscribe.backends import BACKEND_INFO, create_backend
from voxscribe.config import SettingsStore
from voxscribe.diarization import assign_speakers
from voxscribe.exports import write_exports
from voxscribe.preprocessing import preprocess_audio
from voxscribe.tasks import TaskStore


def parse_args():
    parser = argparse.ArgumentParser(prog="voxscribe", description="VoxScribe 本地语音识别 CLI")
    parser.add_argument("input", nargs="?", help="音频或视频文件")
    parser.add_argument("-o", "--output", help="输出目录")
    parser.add_argument("-b", "--backend", choices=sorted(BACKEND_INFO), help="识别后端")
    parser.add_argument("-l", "--language", default="auto", help="auto、Chinese 或 English")
    parser.add_argument("-f", "--format", action="append", choices=["txt", "srt", "vtt", "json"])
    parser.add_argument("--no-preprocess", action="store_true", help="禁用音频预处理")
    parser.add_argument("--speakers", type=int, default=None, help="说话人数；0 为自动")
    parser.add_argument("--list-models", action="store_true", help="列出可用后端")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_models:
        for name, info in BACKEND_INFO.items():
            print(f"{name}\t{info['label']}\t{info['default_path']}")
        return 0
    if not args.input:
        print("错误：请提供输入文件", file=sys.stderr)
        return 2
    source = Path(args.input).resolve()
    if not source.is_file():
        print(f"错误：文件不存在：{source}", file=sys.stderr)
        return 2
    settings = SettingsStore(ROOT / "config" / "settings.json")
    backend_name = args.backend or settings.get("model", "backend", "qwen3_asr")
    model_path = settings.get("model", f"{backend_name}_path", BACKEND_INFO[backend_name]["default_path"])
    output_dir = Path(args.output or settings.get("folder_watch", "output_folder"))
    formats = args.format or settings.get("folder_watch", "export_formats", ["txt"])
    tasks = TaskStore(ROOT / "data" / "tasks.db")
    task_id = tasks.start(source, "cli", backend_name)
    try:
        audio_input = str(source)
        mode = "off" if args.no_preprocess else settings.get("audio_processing", "mode", "noise_reduce")
        if mode != "off":
            print(f"音频预处理：{mode}")
            audio_input = preprocess_audio(source, mode, settings.get("audio_processing", "demucs_model", "htdemucs"))
        print(f"加载模型：{BACKEND_INFO[backend_name]['label']}")
        backend = create_backend(
            backend_name,
            model_path,
            {"command_template": settings.get("model", "external_cli_command", "")},
        )
        language = None if args.language == "auto" else args.language
        result = backend.transcribe_result(audio_input, language)
        speaker_count = args.speakers
        if speaker_count is None and settings.get("audio_processing", "speaker_identification", False):
            speaker_count = settings.get("audio_processing", "speaker_count", 0)
        if speaker_count is not None:
            result = assign_speakers(result, audio_input, speaker_count)
        outputs = write_exports(result, source, output_dir, formats, BACKEND_INFO[backend_name]["label"])
        tasks.complete(task_id, outputs, result)
        print(result.text)
        for output in outputs:
            print(f"已输出：{output}")
        return 0
    except Exception as exc:
        tasks.fail(task_id, exc)
        print(f"转写失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

