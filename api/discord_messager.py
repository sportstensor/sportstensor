import discord
import asyncio
import sys
from api.config import DISCORD_TOKEN


async def send_message(channel_id: int, message: str):
    """
    Sends a message to a specified Discord channel.
    
    Args:
        channel_id (int): The Discord channel ID where the message should be sent.
        message (str): The message to be sent.
    """
    intents = discord.Intents.default()
    client = discord.Client(intents=intents)

    async def on_ready():
        channel = client.get_channel(channel_id)
        if channel:
            await channel.send(message)
        await client.close()  # Close bot after sending message

    client.event(on_ready)
    await client.start(DISCORD_TOKEN)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python discord_messager.py <channel_id> <message>")
        sys.exit(1)

    channel_id = int(sys.argv[1])
    message = " ".join(sys.argv[2:])  # Join message parts

    asyncio.run(send_message(channel_id, message))
