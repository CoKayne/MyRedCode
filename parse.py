import argparse
from pathlib import Path


MODEL_RESPONSE_MARKER = "=== Model Response ==="
CLOSE_TAG = "</think>"
OPEN_TAG = "<think>"


def extract_reasoning_text(log_text: str) -> str:
    marker_index = log_text.find(MODEL_RESPONSE_MARKER)
    if marker_index == -1:
        return log_text.strip()

    segment = log_text[marker_index + len(MODEL_RESPONSE_MARKER):]
    open_index = segment.find(OPEN_TAG)
    start_index = open_index if open_index != -1 else 0
    close_index = segment.find(CLOSE_TAG)

    if close_index != -1 and close_index > start_index:
        return segment[start_index : close_index].strip()
    return segment[start_index:].strip()


def convert_logs(logs_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for log_path in sorted(logs_dir.glob("*.log")):
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        reasoning_text = extract_reasoning_text(log_text)
        output_name = log_path.name.replace(".py", "").replace(".log", "") + ".txt"
        output_path = output_dir / output_name
        output_path.write_text(reasoning_text, encoding="utf-8")
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Parse logs and extract reasoning blocks.")
    parser.add_argument("--logs_dir", default="logs", help="Directory containing log files")
    parser.add_argument("--output_dir", default="reasoning", help="Directory to store extracted reasoning text files")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.is_dir():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    output_dir = Path(args.output_dir)
    total = convert_logs(logs_dir, output_dir)
    print(f"Extracted reasoning from {total} log files into {output_dir}")


if __name__ == "__main__":
    main()
