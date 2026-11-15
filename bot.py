"""
VishCord - Discord voice-phishing (vishing) security bot.

Captures per-user voice audio in real time, converts it in memory to
16 kHz mono float32 (the format speech / audio-classification models expect),
and provides moderation hooks (server mute) for autonomous mitigation.

Voice stack (Discord requires DAVE end-to-end encryption on voice since 2026-03):
  - discord.py (zacker150 fork, adds the hooks the receive extension needs)
  - discord-ext-voice-recv (zacker150 fork, decrypts DAVE audio on receive)
  - davey >=0.1.6,<0.2 (DAVE protocol), PyNaCl (transport encryption)
Install: pip install "git+https://github.com/zacker150/discord-ext-voice-recv"
"""

import asyncio
import os
import threading

import discord
import numpy as np
from discord.ext import commands, voice_recv
from dotenv import load_dotenv
from scipy.signal import resample_poly

# ---------------------------------------------------------------------------
# 1. Environment & Authentication
# ---------------------------------------------------------------------------
load_dotenv()


def get_token() -> str:
    """Read the bot token strictly from DISCORD_BOT_TOKEN."""
    token = (os.getenv("DISCORD_BOT_TOKEN") or "").strip()
    if not token:
        raise ValueError(
            "DISCORD_BOT_TOKEN is missing or empty. Add it to your .env file."
        )
    return token


# ---------------------------------------------------------------------------
# 2. Gateway Intents
# ---------------------------------------------------------------------------
# Server permissions the bot role needs (grant them in the invite URL / role settings):
#   View Channels, Send Messages, Embed Links -> text alerts & embed reports
#   Connect, Speak                            -> join voice & play audio warnings
#   Mute Members                              -> autonomous "Server Mute" mitigation
intents = discord.Intents.default()
intents.message_content = True  # read "!" prefix commands
intents.members = True          # resolve voice members, apply server mutes
intents.voice_states = True     # track users joining / leaving voice channels

bot = commands.Bot(command_prefix="!", intents=intents)

# ---------------------------------------------------------------------------
# 3. Audio constants
# ---------------------------------------------------------------------------
SOURCE_RATE = 48_000            # Discord decodes Opus to 48 kHz
TARGET_RATE = 16_000            # model-friendly rate
CHANNELS = 2                    # Discord PCM is interleaved stereo
WINDOW_SECONDS = 2
WINDOW_SAMPLES = TARGET_RATE * WINDOW_SECONDS  # 32,000 samples @ 16 kHz


class VishCordAudioSink(voice_recv.AudioSink):
    """
    In-memory, per-user audio sink.

    Nothing is written to disk: every PCM frame is converted to 16 kHz mono
    float32 and kept only in RAM until a 2-second analysis window is filled.
    """

    def __init__(self):
        super().__init__()
        self.user_buffers: dict[int, np.ndarray] = {}
        # write() runs on the receive thread; commands run on the event loop.
        self._lock = threading.Lock()

    def wants_opus(self) -> bool:
        """False -> the extension decodes Opus and hands us raw PCM."""
        return False

    # -- DSP pipeline -------------------------------------------------------
    @staticmethod
    def _pcm_to_mono_16k(pcm: bytes) -> np.ndarray:
        """48 kHz stereo int16 PCM -> 16 kHz mono float32 in [-1.0, 1.0]."""
        # 1) int16 -> float32, normalized to [-1.0, 1.0)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

        # 2) Stereo -> Mono: de-interleave [L, R, L, R, ...] and average channels
        usable = len(audio) - (len(audio) % CHANNELS)  # guard against partial frames
        mono = audio[:usable].reshape(-1, CHANNELS).mean(axis=1)

        # 3) Resample 48 kHz -> 16 kHz (exact 1:3 ratio, polyphase anti-aliasing filter)
        return resample_poly(mono, up=1, down=3).astype(np.float32)

    def write(self, user: discord.Member | discord.User | None, data: voice_recv.VoiceData):
        """Called on the receive thread for every decoded 20 ms frame of a speaker."""
        if user is None or not data.pcm:  # unknown SSRC owner or empty frame
            return
        samples = self._pcm_to_mono_16k(data.pcm)

        with self._lock:
            # 4) Accumulate per user ID
            buf = self.user_buffers.get(user.id)
            buf = samples if buf is None else np.concatenate((buf, samples))

            # Trigger hook: every full 2 s window is handed off, the remainder slides forward
            while len(buf) >= WINDOW_SAMPLES:
                window, buf = buf[:WINDOW_SAMPLES], buf[WINDOW_SAMPLES:]
                self.on_window(user.id, window)

            self.user_buffers[user.id] = buf

    def on_window(self, user_id: int, window: np.ndarray):
        """
        Analysis hook for one 2-second, 16 kHz mono window.
        Sanity check for now; later: feed `window` to the vishing detector.
        Keep this fast - it runs on the audio thread.
        """
        print(f"[AUDIO] user={user_id} window={len(window) / TARGET_RATE:.2f}s "
              f"peak={float(np.abs(window).max()):.3f}")

    def drop_user(self, user_id: int):
        """Wipe a single user's buffer (e.g. when they leave the channel)."""
        with self._lock:
            self.user_buffers.pop(user_id, None)

    def cleanup(self):
        """Wipe all audio from RAM. Called by the extension when listening stops."""
        lock = getattr(self, "_lock", None)  # also runs from __del__, maybe half-built
        if lock is None:
            return
        with lock:
            self.user_buffers.clear()


# ---------------------------------------------------------------------------
# 4. Helpers
# ---------------------------------------------------------------------------
def status_embed(title: str, description: str, color: discord.Color) -> discord.Embed:
    """Build a VishCord-branded status embed (requires Embed Links)."""
    embed = discord.Embed(title=title, description=description, color=color)
    embed.set_footer(text="VishCord • audio processed in memory only")
    return embed


def on_listening_finished(error: Exception | None):
    """Called (from the receive thread) when listening ends; error is None if clean."""
    if error:
        print(f"[VOICE] Listening stopped with error: {error!r}")
    else:
        print("[VOICE] Listening stopped, buffers cleared.")


def get_sink(vc: voice_recv.VoiceRecvClient | None) -> VishCordAudioSink | None:
    """Return the active VishCord sink on a voice client, if any."""
    if vc and vc.is_listening() and isinstance(vc.sink, VishCordAudioSink):
        return vc.sink
    return None


# ---------------------------------------------------------------------------
# 5. Events
# ---------------------------------------------------------------------------
@bot.event
async def on_ready():
    print(f"[READY] VishCord online as {bot.user} (id={bot.user.id})")


@bot.event
async def on_voice_state_update(member: discord.Member, before, after):
    """Drop a user's buffered audio as soon as they leave the bot's channel."""
    vc = member.guild.voice_client
    sink = get_sink(vc)
    if sink and before.channel == vc.channel and after.channel != vc.channel:
        sink.drop_user(member.id)


# ---------------------------------------------------------------------------
# 6. Commands
# ---------------------------------------------------------------------------
@bot.command(name="join")
@commands.guild_only()
async def join(ctx: commands.Context):
    """Join the caller's voice channel and start in-memory monitoring."""
    if not ctx.author.voice or not ctx.author.voice.channel:
        return await ctx.send("❌ You must be in a voice channel first.")

    channel = ctx.author.voice.channel
    perms = channel.permissions_for(ctx.guild.me)
    if not (perms.connect and perms.speak):
        return await ctx.send(f"❌ I need **Connect** and **Speak** in {channel.mention}.")

    try:
        vc = ctx.voice_client
        if vc and not isinstance(vc, voice_recv.VoiceRecvClient):
            await vc.disconnect(force=True)  # plain client can't receive audio
            vc = None

        if vc:
            if vc.is_listening():
                vc.stop_listening()  # sink.cleanup() wipes RAM
            if vc.channel != channel:
                await vc.move_to(channel)
        else:
            # cls=VoiceRecvClient is what enables (DAVE-decrypted) audio receive
            vc = await channel.connect(cls=voice_recv.VoiceRecvClient, timeout=30)

        vc.listen(VishCordAudioSink(), after=on_listening_finished)
    except discord.Forbidden:
        return await ctx.send("❌ Missing permission to join that voice channel.")
    except asyncio.TimeoutError:
        return await ctx.send("❌ Timed out connecting to voice. Try again.")
    except discord.ClientException as e:
        return await ctx.send(f"❌ Could not start monitoring: {e}")

    embed = status_embed(
        "🛡️ VishCord Active",
        f"Monitoring **{channel.name}** for voice-phishing threats.",
        discord.Color.green(),
    )
    try:
        await ctx.send(embed=embed)
    except discord.Forbidden:  # no Embed Links -> plain text fallback
        await ctx.send(f"🛡️ VishCord active in {channel.name}.")


@bot.command(name="leave")
@commands.guild_only()
async def leave(ctx: commands.Context):
    """Stop listening, wipe buffers and disconnect."""
    vc = ctx.voice_client
    if not vc:
        return await ctx.send("ℹ️ I'm not connected to a voice channel.")

    if isinstance(vc, voice_recv.VoiceRecvClient) and vc.is_listening():
        vc.stop_listening()  # calls VishCordAudioSink.cleanup() -> RAM wiped
    await vc.disconnect()
    await ctx.send("👋 Disconnected. All audio buffers wiped from memory.")


@bot.command(name="test_mute")
@commands.guild_only()
async def test_mute(ctx: commands.Context, member: discord.Member):
    """Verify Mute Members permission + role hierarchy by server-muting a member."""
    me = ctx.guild.me
    if not me.guild_permissions.mute_members:
        return await ctx.send("❌ I lack the **Mute Members** permission.")
    if member.top_role >= me.top_role:
        return await ctx.send(
            f"❌ {member.mention}'s top role is at or above mine - move my role higher."
        )
    if not member.voice:
        return await ctx.send(f"❌ {member.mention} must be in a voice channel to be muted.")

    try:
        await member.edit(mute=True, reason="VishCord test_mute")
        await ctx.send(f"🔇 {member.mention} server-muted. Permissions OK.")
    except discord.Forbidden:
        await ctx.send(f"❌ Forbidden: cannot mute {member.mention} (owner/hierarchy).")
    except discord.HTTPException as e:
        await ctx.send(f"❌ Mute failed: {e}")


@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    """Friendly messages for common command errors."""
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"❌ Usage: `!{ctx.command.name} {ctx.command.signature}`")
    elif isinstance(error, commands.MemberNotFound):
        await ctx.send("❌ Member not found.")
    elif isinstance(error, commands.NoPrivateMessage):
        await ctx.send("❌ This command only works in a server.")
    elif isinstance(error, commands.CommandNotFound):
        return
    else:
        raise error


if __name__ == "__main__":
    bot.run(get_token())
