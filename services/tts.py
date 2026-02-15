"""
JARVIS TTS (with built-in rephraser)
System text → JARVIS-style speech
"""

import asyncio
import edge_tts
import os
import tempfile
import platform
import time

from services.rephraser import rephrase


class JarvisTTS:
    def __init__(self):
        self.voice = "en-GB-RyanNeural"
        self.rate = "-5%"
        self.pitch = "+1Hz"

    async def _generate_and_play(self, text: str):
        # Temp file only for playback
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp:
            audio_path = temp.name

        try:
            communicate = edge_tts.Communicate(
                text,
                voice=self.voice,
                rate=self.rate,
                pitch=self.pitch,
            )
            await communicate.save(audio_path)
            self._play(audio_path)
        finally:
            try:
                os.remove(audio_path)
            except:
                pass

    async def say(self, system_text: str):
        """
        Rephrase system text into JARVIS voice, then speak it.
        """
        # Step 1: Rephrase (LLM)
        spoken_text = await rephrase(system_text)

        print(f"🤖 JARVIS: {spoken_text}")

        # Step 2: Speak (TTS)
        await self._generate_and_play(spoken_text)

    def _play(self, file: str):
        try:
            if platform.system() == "Windows":
                try:
                    from pygame import mixer
                    mixer.init()
                    mixer.music.load(file)
                    mixer.music.play()
                    while mixer.music.get_busy():
                        time.sleep(0.05)
                    mixer.quit()
                except Exception:
                    os.startfile(file)
            else:
                import subprocess
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", file]
                )
        except Exception as e:
            print(f"⚠️  Audio play error: {e}")
