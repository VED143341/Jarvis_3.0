import asyncio
import time
from datetime import datetime

from services.rephraser import rephrase


async def timed_call(label: str, text: str):
    start = time.perf_counter()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] START {label}")

    result = await rephrase(text)

    duration = time.perf_counter() - start
    print(f"[{datetime.now().strftime('%H:%M:%S')}] END   {label} | {duration:.2f}s")
    print(f"→ {label} output: {result}\n")


async def main():
    overall_start = time.perf_counter()

    print("=== Sequential async calls ===\n")

    await timed_call("A", "everything is setup")
    await timed_call("B", "the system is ready")
    await timed_call("C", "waiting for the next command")

    print("\n=== Queued async calls (gather) ===\n")

    await asyncio.gather(
        timed_call("D", "initialization complete"),
        timed_call("E", "all services are online"),
        timed_call("F", "standing by"),
    )

    total = time.perf_counter() - overall_start
    print(f"\nTOTAL elapsed time: {total:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
