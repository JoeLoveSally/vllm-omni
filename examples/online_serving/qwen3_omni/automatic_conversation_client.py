# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Reference client for Qwen3-Omni automatic multi-turn Realtime conversation.

This client intentionally uses only the public WebSocket Realtime protocol.
It does not depend on vLLM-Omni internals, so the same acceptance flow can be
run against the current serving path and the future Unified Full-duplex path.

Acceptance properties:

* one WebSocket / one session for all user turns;
* PCM16 mono 16 kHz audio is sent at its real media rate;
* Server VAD owns endpointing;
* the client never sends ``input_audio_buffer.commit``;
* the client never sends ``response.create``;
* every completed speech turn creates exactly one response;
* input-item and response identities are checked across turns;
* event ordering is checked;
* optional semantic assertions verify cross-turn history.

Example:

    python examples/online_serving/qwen3_omni/automatic_conversation_client.py \
        --url ws://127.0.0.1:8091/v1/realtime \
        --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --turn-wav ./turn1.wav \
        --turn-wav ./turn2.wav \
        --instructions "Answer briefly and remember information from earlier turns." \
        --expect-substring '2:蓝鲸七号' \
        --output-dir ./m1-out
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import websockets

INPUT_SAMPLE_RATE = 16_000
INPUT_SAMPLE_WIDTH = 2
DEFAULT_OUTPUT_SAMPLE_RATE = 24_000

FORBIDDEN_CLIENT_EVENTS = {
    "input_audio_buffer.commit",
    "response.create",
}

REQUIRED_TURN_SEQUENCE = [
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.committed",
    "response.created",
    "response.output_audio.delta",
    "response.output_audio.done",
    "response.done",
]


@dataclass
class EventJournal:
    path: Path | None
    started_monotonic: float = field(default_factory=time.monotonic)

    def write(
        self,
        *,
        direction: str,
        event: dict[str, Any],
        turn: int | None = None,
    ) -> None:
        if self.path is None:
            return

        record = {
            "elapsed_s": round(time.monotonic() - self.started_monotonic, 6),
            "unix_s": time.time(),
            "direction": direction,
            "turn": turn,
            "event": _event_for_log(event),
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class TurnResult:
    turn: int
    events: list[dict[str, Any]]
    input_item_id: str
    response_id: str
    text: str
    output_pcm: bytes
    output_sample_rate: int

    def summary(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "input_item_id": self.input_item_id,
            "response_id": self.response_id,
            "text": self.text,
            "output_pcm_bytes": len(self.output_pcm),
            "output_sample_rate": self.output_sample_rate,
            "event_types": [event.get("type") for event in self.events],
        }


def _event_for_log(event: dict[str, Any]) -> dict[str, Any]:
    """Keep protocol metadata while avoiding huge base64 blobs in JSONL."""

    result = dict(event)

    audio = result.get("audio")
    if isinstance(audio, str):
        result["audio"] = f"<base64 chars={len(audio)}>"

    delta = result.get("delta")
    if result.get("type") == "response.output_audio.delta" and isinstance(delta, str):
        result["delta"] = f"<base64 chars={len(delta)}>"

    return result


def _read_pcm16_mono_16k(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono WAV, got {wf.getnchannels()} channels")
        if wf.getsampwidth() != INPUT_SAMPLE_WIDTH:
            raise ValueError(f"{path}: expected PCM16, sample width={wf.getsampwidth()}")
        if wf.getframerate() != INPUT_SAMPLE_RATE:
            raise ValueError(f"{path}: expected 16 kHz, got {wf.getframerate()} Hz")
        if wf.getcomptype() != "NONE":
            raise ValueError(f"{path}: expected uncompressed PCM, got {wf.getcomptype()!r}")

        pcm = wf.readframes(wf.getnframes())

    if not pcm:
        raise ValueError(f"{path}: WAV contains no audio")

    return pcm


def _write_pcm16_wav(
    path: Path,
    pcm: bytes,
    *,
    sample_rate: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


async def _send_event(
    ws,
    event: dict[str, Any],
    *,
    journal: EventJournal,
    sent_event_types: list[str],
    turn: int | None = None,
) -> None:
    event_type = str(event.get("type", ""))

    if event_type in FORBIDDEN_CLIENT_EVENTS:
        raise AssertionError(f"M1 client must never send forbidden event: {event_type}")

    sent_event_types.append(event_type)
    journal.write(direction="client->server", event=event, turn=turn)
    await ws.send(json.dumps(event, ensure_ascii=False))


async def _recv_event(
    ws,
    *,
    journal: EventJournal,
    timeout_s: float,
    turn: int | None = None,
) -> dict[str, Any]:
    while True:
        message = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
        if isinstance(message, bytes):
            continue

        event = json.loads(message)
        journal.write(direction="server->client", event=event, turn=turn)

        if event.get("type") == "error":
            raise RuntimeError(f"Server error: {event}")

        return event


async def _wait_for_session_updated(
    ws,
    *,
    journal: EventJournal,
    timeout_s: float,
    model: str,
    instructions: str | None,
) -> dict[str, Any]:
    """Wait until the server confirms the effective session configuration."""

    while True:
        event = await _recv_event(
            ws,
            journal=journal,
            timeout_s=timeout_s,
        )

        if event.get("type") != "session.updated":
            continue

        session = event.get("session")
        if not isinstance(session, dict):
            raise AssertionError(f"session.updated missing session object: {event}")

        effective_model = session.get("model")
        if effective_model is not None and effective_model != model:
            raise AssertionError(f"Unexpected effective model: {effective_model!r} != {model!r}")

        if instructions is not None:
            effective_instructions = session.get("instructions")
            if effective_instructions != instructions:
                raise AssertionError(
                    f"Server did not preserve session instructions: {effective_instructions!r} != {instructions!r}"
                )

        turn_detection = session.get("audio", {}).get("input", {}).get("turn_detection", {})
        if turn_detection.get("type") != "server_vad":
            raise AssertionError(f"Expected server_vad, got {turn_detection!r}")

        return session


async def _append_pcm_realtime(
    ws,
    pcm: bytes,
    *,
    chunk_ms: int,
    journal: EventJournal,
    sent_event_types: list[str],
    turn: int,
) -> None:
    """Append PCM while pacing against the audio media clock.

    This uses cumulative media duration rather than sleeping a fixed amount
    after each write, which avoids accumulating scheduler/send overhead.
    """

    if chunk_ms <= 0:
        raise ValueError("chunk_ms must be > 0")

    bytes_per_second = INPUT_SAMPLE_RATE * INPUT_SAMPLE_WIDTH
    chunk_bytes = max(bytes_per_second * chunk_ms // 1000, 2)
    chunk_bytes -= chunk_bytes % INPUT_SAMPLE_WIDTH

    loop = asyncio.get_running_loop()
    started = loop.time()
    bytes_sent = 0

    for offset in range(0, len(pcm), chunk_bytes):
        chunk = pcm[offset : offset + chunk_bytes]

        await _send_event(
            ws,
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("ascii"),
            },
            journal=journal,
            sent_event_types=sent_event_types,
            turn=turn,
        )

        bytes_sent += len(chunk)
        media_elapsed = bytes_sent / bytes_per_second
        sleep_for = started + media_elapsed - loop.time()

        if sleep_for > 0:
            await asyncio.sleep(sleep_for)


async def _send_turn(
    ws,
    *,
    pcm: bytes,
    chunk_ms: int,
    trailing_silence_ms: int,
    journal: EventJournal,
    sent_event_types: list[str],
    turn: int,
) -> None:
    await _append_pcm_realtime(
        ws,
        pcm,
        chunk_ms=chunk_ms,
        journal=journal,
        sent_event_types=sent_event_types,
        turn=turn,
    )

    if trailing_silence_ms <= 0:
        return

    silence = bytes(INPUT_SAMPLE_RATE * INPUT_SAMPLE_WIDTH * trailing_silence_ms // 1000)

    await _append_pcm_realtime(
        ws,
        silence,
        chunk_ms=chunk_ms,
        journal=journal,
        sent_event_types=sent_event_types,
        turn=turn,
    )


def _extract_response_text(events: list[dict[str, Any]]) -> str:
    """Extract the best textual representation of one assistant response."""

    final_candidates: dict[str, str] = {}
    delta_channels: dict[str, list[str]] = {
        "response.output_text.delta": [],
        "transcription.delta": [],
        "response.output_audio_transcript.delta": [],
    }

    for event in events:
        event_type = event.get("type")

        if event_type in delta_channels:
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                delta_channels[event_type].append(delta)

        elif event_type == "response.output_text.done":
            text = event.get("text")
            if isinstance(text, str) and text:
                final_candidates["response.output_text.done"] = text

        elif event_type == "transcription.done":
            text = event.get("text")
            if isinstance(text, str) and text:
                final_candidates["transcription.done"] = text

        elif event_type == "response.output_audio_transcript.done":
            text = event.get("transcript")
            if isinstance(text, str) and text:
                final_candidates["response.output_audio_transcript.done"] = text

    for event_type in (
        "response.output_text.done",
        "transcription.done",
        "response.output_audio_transcript.done",
    ):
        if event_type in final_candidates:
            return final_candidates[event_type].strip()

    for event_type in (
        "response.output_text.delta",
        "transcription.delta",
        "response.output_audio_transcript.delta",
    ):
        text = "".join(delta_channels[event_type]).strip()
        if text:
            return text

    return ""


def _collect_output_audio(
    events: list[dict[str, Any]],
) -> tuple[bytes, int]:
    chunks: list[bytes] = []
    sample_rate = DEFAULT_OUTPUT_SAMPLE_RATE

    for event in events:
        if event.get("type") != "response.output_audio.delta":
            continue

        sr = event.get("sample_rate_hz")
        if isinstance(sr, int) and sr > 0:
            sample_rate = sr

        encoded = event.get("delta")
        if not isinstance(encoded, str):
            encoded = event.get("audio")

        if isinstance(encoded, str) and encoded:
            chunks.append(base64.b64decode(encoded))

    return b"".join(chunks), sample_rate


def _assert_turn_protocol(
    events: list[dict[str, Any]],
    *,
    turn: int,
) -> tuple[str, str]:
    event_types = [str(event.get("type")) for event in events]

    for event_type in REQUIRED_TURN_SEQUENCE:
        count = event_types.count(event_type)

        if event_type == "response.output_audio.delta":
            if count < 1:
                raise AssertionError(f"Turn {turn}: expected >=1 {event_type}, got {event_types}")
        elif count != 1:
            raise AssertionError(f"Turn {turn}: expected exactly one {event_type}, got {count}: {event_types}")

    positions = [event_types.index(event_type) for event_type in REQUIRED_TURN_SEQUENCE]
    if positions != sorted(positions):
        raise AssertionError(f"Turn {turn}: invalid event ordering: {event_types}")

    started = next(event for event in events if event.get("type") == "input_audio_buffer.speech_started")
    stopped = next(event for event in events if event.get("type") == "input_audio_buffer.speech_stopped")
    committed = next(event for event in events if event.get("type") == "input_audio_buffer.committed")
    created = next(event for event in events if event.get("type") == "response.created")["response"]
    done = next(event for event in events if event.get("type") == "response.done")["response"]

    input_item_id = committed["item_id"]

    if not (started.get("item_id") == stopped.get("item_id") == input_item_id):
        raise AssertionError(
            f"Turn {turn}: speech/commit item identity mismatch: "
            f"started={started.get('item_id')!r}, "
            f"stopped={stopped.get('item_id')!r}, "
            f"committed={input_item_id!r}"
        )

    response_id = created["id"]
    if response_id != done.get("id"):
        raise AssertionError(f"Turn {turn}: response.created/done ID mismatch: {response_id!r} != {done.get('id')!r}")

    if done.get("status") != "completed":
        raise AssertionError(f"Turn {turn}: response ended with status={done.get('status')!r}")

    return input_item_id, response_id


async def _receive_turn(
    ws,
    *,
    turn: int,
    journal: EventJournal,
    timeout_s: float,
) -> TurnResult:
    events: list[dict[str, Any]] = []

    while True:
        event = await _recv_event(
            ws,
            journal=journal,
            timeout_s=timeout_s,
            turn=turn,
        )
        events.append(event)

        if event.get("type") == "response.done":
            break

    input_item_id, response_id = _assert_turn_protocol(
        events,
        turn=turn,
    )
    text = _extract_response_text(events)
    output_pcm, output_sample_rate = _collect_output_audio(events)

    if not output_pcm:
        raise AssertionError(f"Turn {turn}: response contained no output audio")

    return TurnResult(
        turn=turn,
        events=events,
        input_item_id=input_item_id,
        response_id=response_id,
        text=text,
        output_pcm=output_pcm,
        output_sample_rate=output_sample_rate,
    )


def _parse_expectations(
    raw_expectations: list[str],
) -> list[tuple[int, str]]:
    expectations: list[tuple[int, str]] = []

    for value in raw_expectations:
        try:
            turn_text, expected = value.split(":", 1)
            turn = int(turn_text)
        except ValueError as exc:
            raise ValueError("--expect-substring must use TURN:TEXT, for example '2:蓝鲸七号'") from exc

        if turn <= 0 or not expected:
            raise ValueError(f"Invalid --expect-substring value: {value!r}")

        expectations.append((turn, expected))

    return expectations


def _assert_semantics(
    results: list[TurnResult],
    expectations: list[tuple[int, str]],
) -> None:
    by_turn = {result.turn: result for result in results}

    for turn, expected in expectations:
        if turn not in by_turn:
            raise AssertionError(f"Semantic assertion references missing turn {turn}")

        actual = by_turn[turn].text
        if not actual:
            raise AssertionError(f"Turn {turn}: no textual response is available; cannot assert substring {expected!r}")

        if expected.casefold() not in actual.casefold():
            raise AssertionError(f"Turn {turn}: expected substring {expected!r} not found in response: {actual!r}")


async def run(args: argparse.Namespace) -> None:
    if len(args.turn_wav) < 2:
        raise ValueError("M1 automatic-conversation acceptance requires at least two --turn-wav inputs")

    expectations = _parse_expectations(args.expect_substring)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    event_log = args.event_log or output_dir / "events.jsonl"
    event_log.unlink(missing_ok=True)

    journal = EventJournal(event_log)
    sent_event_types: list[str] = []
    results: list[TurnResult] = []

    turn_pcm = [_read_pcm16_mono_16k(path) for path in args.turn_wav]

    session_config: dict[str, Any] = {
        "model": args.model,
        "audio": {
            "input": {
                "format": {
                    "type": "audio/pcm",
                    "rate": INPUT_SAMPLE_RATE,
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": args.vad_threshold,
                    "prefix_padding_ms": args.prefix_padding_ms,
                    "silence_duration_ms": args.vad_silence_ms,
                    "create_response": True,
                    "interrupt_response": False,
                },
            }
        },
    }

    if args.instructions is not None:
        session_config["instructions"] = args.instructions

    print(f"[session] connecting to {args.url}")

    async with websockets.connect(
        args.url,
        max_size=64 * 1024 * 1024,
    ) as ws:
        await _send_event(
            ws,
            {
                "type": "session.update",
                "session": session_config,
            },
            journal=journal,
            sent_event_types=sent_event_types,
        )

        effective_session = await _wait_for_session_updated(
            ws,
            journal=journal,
            timeout_s=args.timeout_s,
            model=args.model,
            instructions=args.instructions,
        )

        print("[session] server confirmed server_vad" + (" and instructions" if args.instructions is not None else ""))

        for turn, pcm in enumerate(turn_pcm, start=1):
            wav_path = args.turn_wav[turn - 1]
            duration_s = len(pcm) / INPUT_SAMPLE_WIDTH / INPUT_SAMPLE_RATE

            print(f"[turn {turn}] sending {wav_path} ({duration_s:.3f}s) in realtime")

            await _send_turn(
                ws,
                pcm=pcm,
                chunk_ms=args.chunk_ms,
                trailing_silence_ms=args.trailing_silence_ms,
                journal=journal,
                sent_event_types=sent_event_types,
                turn=turn,
            )

            result = await _receive_turn(
                ws,
                turn=turn,
                journal=journal,
                timeout_s=args.timeout_s,
            )
            results.append(result)

            output_wav = output_dir / f"turn_{turn:02d}_response.wav"
            _write_pcm16_wav(
                output_wav,
                result.output_pcm,
                sample_rate=result.output_sample_rate,
            )

            print(f"[turn {turn}] input_item_id={result.input_item_id} response_id={result.response_id}")
            print(f"[turn {turn}] text={result.text!r}")
            print(f"[turn {turn}] audio={output_wav}")

    forbidden_seen = FORBIDDEN_CLIENT_EVENTS.intersection(sent_event_types)
    if forbidden_seen:
        raise AssertionError(f"Forbidden client events were sent: {sorted(forbidden_seen)}")

    input_item_ids = [result.input_item_id for result in results]
    response_ids = [result.response_id for result in results]

    if len(set(input_item_ids)) != len(input_item_ids):
        raise AssertionError(f"Input item IDs are not unique: {input_item_ids}")

    if len(set(response_ids)) != len(response_ids):
        raise AssertionError(f"Response IDs are not unique: {response_ids}")

    _assert_semantics(results, expectations)

    summary = {
        "url": args.url,
        "model": args.model,
        "instructions": args.instructions,
        "effective_session": effective_session,
        "client_sent_event_types": sent_event_types,
        "forbidden_client_events_sent": sorted(forbidden_seen),
        "turns": [result.summary() for result in results],
        "semantic_expectations": [
            {
                "turn": turn,
                "substring": expected,
            }
            for turn, expected in expectations
        ],
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("[PASS] M1 automatic-conversation acceptance passed")
    print(f"[PASS] turns={len(results)}")
    print("[PASS] client commit events=0")
    print("[PASS] client response.create events=0")
    print(f"[PASS] event log: {event_log}")
    print(f"[PASS] summary:   {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Transport-independent Qwen3-Omni M1 automatic-conversation Realtime acceptance client")
    )
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8091/v1/realtime",
        help=(
            "Realtime WebSocket URL. The client uses the URL exactly as provided, "
            "so it can target any compatible Realtime endpoint."
        ),
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    )
    parser.add_argument(
        "--turn-wav",
        action="append",
        type=Path,
        required=True,
        help=("User-turn WAV; repeat at least twice. Each file must be mono PCM16 16 kHz."),
    )
    parser.add_argument(
        "--instructions",
        default=None,
        help="Session instructions that must survive session.update.",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=100,
        help="PCM append chunk duration; chunks are paced by media time.",
    )
    parser.add_argument(
        "--trailing-silence-ms",
        type=int,
        default=1200,
        help="Realtime-paced silence appended after each user WAV.",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--prefix-padding-ms",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--vad-silence-ms",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=600.0,
    )
    parser.add_argument(
        "--expect-substring",
        action="append",
        default=[],
        help=("Semantic assertion in TURN:TEXT form; repeat if needed. Example: --expect-substring '2:蓝鲸七号'"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("qwen_m1_automatic_conversation"),
    )
    parser.add_argument(
        "--event-log",
        type=Path,
        default=None,
        help="JSONL protocol journal; defaults to OUTPUT_DIR/events.jsonl.",
    )

    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
