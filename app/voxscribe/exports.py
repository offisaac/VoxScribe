from pathlib import Path


def timestamp(seconds, separator=","):
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def write_exports(result, source, output_dir, formats, backend_label):
    source = Path(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    stem = f"{source.stem} - {backend_label}"
    segments = [segment for segment in result.segments if segment.text.strip()]
    for export_format in formats or ["txt"]:
        export_format = export_format.lower()
        output = output_dir / f"{stem}.{export_format}"
        if export_format == "txt":
            content = "\n".join(
                f"{segment.speaker}: {segment.text.strip()}" if segment.speaker else segment.text.strip()
                for segment in segments
            )
        elif export_format == "srt":
            blocks = []
            for index, segment in enumerate(segments, 1):
                text = f"{segment.speaker}: {segment.text.strip()}" if segment.speaker else segment.text.strip()
                blocks.append(f"{index}\n{timestamp(segment.start)} --> {timestamp(segment.end)}\n{text}")
            content = "\n\n".join(blocks) + "\n"
        elif export_format == "vtt":
            blocks = ["WEBVTT"]
            for segment in segments:
                text = f"{segment.speaker}: {segment.text.strip()}" if segment.speaker else segment.text.strip()
                blocks.append(f"{timestamp(segment.start, '.')} --> {timestamp(segment.end, '.')}\n{text}")
            content = "\n\n".join(blocks) + "\n"
        elif export_format == "json":
            content = result.to_json()
        else:
            continue
        output.write_text(content, encoding="utf-8")
        outputs.append(output)
    return outputs

