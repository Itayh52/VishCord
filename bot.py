import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
import numpy as np
from scipy.signal import resample_poly

load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.voice_states = True

bot = commands.Bot(command_prefix="!", intents=intents)


class VishCordAudioSink(discord.sinks.Sink):
    """Captures per-user audio streams and converts 48kHz Stereo PCM to 16kHz Mono in memory."""

    def __init__(self):
        super().__init__()
        self.user_buffers = {}

    def write(self, data: bytes, user: int):
        # Convert raw PCM bytes to normalized float32
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        # Stereo to Mono
        if len(audio) % 2 == 0:
            audio = audio.reshape(-1, 2).mean(axis=1)

        # Resample 48kHz -> 16kHz (1:3 ratio)
        mono_16k = resample_poly(audio, up=1, down=3)

        # Append to user buffer
        buf = self.user_buffers.get(user, np.array([], dtype=np.float32))
        buf = np.concatenate((buf, mono_16k))

        # Sanity check every 2 seconds (32,000 samples)
        if len(buf) >= 32000:
            print(f"[AUDIO] User {user}: captured {len(buf) / 16000:.2f}s (16kHz Mono)")
            buf = np.array([], dtype=np.float32)

        self.user_buffers[user] = buf

    def cleanup(self):
        self.user_buffers.clear()


async def finished_callback(sink: VishCordAudioSink, channel: discord.TextChannel):
    sink.cleanup()
    await channel.send("🛡️ **VishCord:** Monitoring stopped and memory buffers cleared.")


@bot.event
async def on_ready():
    print(f"✅ {bot.user} is online and ready.")


@bot.command()
async def join(ctx: commands.Context):
    """Join voice channel and start monitoring."""
    if not ctx.author.voice:
        return await ctx.send("❌ Join a voice channel first.")

    channel = ctx.author.voice.channel
    if ctx.voice_client:
        await ctx.voice_client.move_to(channel)
    else:
        await channel.connect()

    ctx.voice_client.start_recording(VishCordAudioSink(), finished_callback, ctx.channel)

    embed = discord.Embed(
        title="🛡️ VishCord Active Defense",
        description=f"Monitoring voice channel: **{channel.name}**",
        color=discord.Color.green(),
    )
    await ctx.send(embed=embed)


@bot.command()
async def leave(ctx: commands.Context):
    """Stop monitoring and disconnect."""
    if not ctx.voice_client:
        return await ctx.send("❌ Not connected to a voice channel.")

    ctx.voice_client.stop_recording()
    await ctx.voice_client.disconnect()
    await ctx.send("👋 Disconnected from voice channel.")


@bot.command()
async def test_mute(ctx: commands.Context, member: discord.Member):
    """Verify server mute permissions."""
    try:
        await member.edit(mute=True, reason="VishCord Security Test")
        await ctx.send(f"🔇 Muted {member.mention}.")
    except discord.Forbidden:
        await ctx.send("❌ Missing 'Mute Members' permission or role hierarchy issue.")


if __name__ == "__main__":
    if not TOKEN:
        raise ValueError("DISCORD_TOKEN not found in .env")
    bot.run(TOKEN)