import asyncio
import time
from datetime import datetime

from services.tts import JarvisTTS
from services.rephraser import rephrase


jarvis = JarvisTTS()


async def timed_say(label: str, system_text: str):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] START {label}")
    start = time.perf_counter()

    # 🧠 Rephrase timing
    llm_start = time.perf_counter()
    spoken_text = await rephrase(system_text)
    llm_time = time.perf_counter() - llm_start

    print(f"   🧠 Rephrased ({llm_time:.2f}s): {spoken_text}")

    # 🎧 Speak timing
    tts_start = time.perf_counter()
    await jarvis.say(system_text)  # internally rephrases + speaks
    tts_time = time.perf_counter() - tts_start

    total = time.perf_counter() - start

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] END   {label} | "
        f"LLM: {llm_time:.2f}s | TTS: {tts_time:.2f}s | TOTAL: {total:.2f}s"
    )


async def main():
    overall_start = time.perf_counter()

    messages = [
        "everything is setup",
        "all services are online",
        "initialization complete",
        "background processes are running",
        "waiting for the next command",
        "system resources are within normal limits",
        "no issues detected at this time",
        "standing by",
    ]

    print("=== Sequential JARVIS responses ===")

    for i, msg in enumerate(messages, start=1):
        await timed_say(f"A{i}", msg)

    print("\n=== Queued JARVIS responses (async gather) ===")

    await asyncio.gather(
        *(timed_say(f"B{i}", msg) for i, msg in enumerate(messages, start=1))
    )

    total = time.perf_counter() - overall_start
    print(f"\n🏁 TOTAL elapsed time: {total:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
