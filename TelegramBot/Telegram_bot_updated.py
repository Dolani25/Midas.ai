import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
import aiohttp
import json
from config import TOKEN, ALLOWED_USERS, PASTEBIN_API_KEY
from database import Database
from sniper_bot import SniperBot  # Assuming you have a main class for your sniper bot

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database and sniper bot
db = Database()
sniper_bot = SniperBot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if str(user.id) not in ALLOWED_USERS:
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        return
    
    await update.message.reply_text(
        f"Welcome, {user.mention_html()}! I'm your Solana Memecoin Sniper Bot assistant. "
        "Use /help to see available commands.",
        parse_mode='HTML'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Available commands:\n"
        "/status - Check bot status\n"
        "/set_strategy - Set a new trading strategy\n"
        "/toggle_ai - Toggle between AI and custom strategy\n"
        "/share_strategy - Share your current strategy\n"
        "/import_strategy - Import a strategy from a Pastebin link"
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    status = sniper_bot.get_status()
    await update.message.reply_text(f"Current bot status:\n{status}")

async def set_strategy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Please enter your trading strategy in natural language. For example:\n"
        "'Buy when RSI is less than 30, sell when greater than 70'"
    )
    context.user_data['awaiting_strategy'] = True

async def handle_strategy_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.user_data.get('awaiting_strategy'):
        strategy = update.message.text
        user_id = update.effective_user.id
        
        # Here you would process and validate the strategy
        success = sniper_bot.set_custom_strategy(user_id, strategy)
        
        if success:
            await update.message.reply_text("Strategy set successfully!")
        else:
            await update.message.reply_text("Failed to set strategy. Please try again.")
        
        context.user_data['awaiting_strategy'] = False

async def toggle_ai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    current_mode = sniper_bot.toggle_ai_mode(user_id)
    await update.message.reply_text(f"Trading mode set to: {'AI' if current_mode else 'Custom Strategy'}")

async def share_strategy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    strategy = sniper_bot.get_user_strategy(user_id)
    
    if not strategy:
        await update.message.reply_text("You don't have a custom strategy set.")
        return
    
    pastebin_url = await upload_to_pastebin(strategy)
    if pastebin_url:
        await update.message.reply_text(f"Your strategy has been shared. Here's the link: {pastebin_url}")
    else:
        await update.message.reply_text("Failed to share strategy. Please try again later.")

async def import_strategy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Please enter the Pastebin link of the strategy you want to import.")
    context.user_data['awaiting_pastebin_link'] = True

async def handle_pastebin_link(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.user_data.get('awaiting_pastebin_link'):
        pastebin_link = update.message.text
        strategy = await fetch_from_pastebin(pastebin_link)
        
        if strategy:
            user_id = update.effective_user.id
            success = sniper_bot.set_custom_strategy(user_id, strategy)
            if success:
                await update.message.reply_text("Strategy imported and set successfully!")
            else:
                await update.message.reply_text("Failed to set imported strategy. Please try again.")
        else:
            await update.message.reply_text("Failed to fetch strategy from Pastebin. Please check the link and try again.")
        
        context.user_data['awaiting_pastebin_link'] = False

async def upload_to_pastebin(text: str) -> str:
    async with aiohttp.ClientSession() as session:
        payload = {
            'api_dev_key': PASTEBIN_API_KEY,
            'api_option': 'paste',
            'api_paste_code': text,
            'api_paste_private': '1',  # Unlisted paste
            'api_paste_name': 'Solana Memecoin Sniper Strategy',
            'api_paste_expire_date': '1W'  # Expire after 1 week
        }
        async with session.post('https://pastebin.com/api/api_post.php', data=payload) as response:
            if response.status == 200:
                return await response.text()
    return None

async def fetch_from_pastebin(url: str) -> str:
    # Extract paste ID from URL
    paste_id = url.split('/')[-1]
    raw_url = f'https://pastebin.com/raw/{paste_id}'
    
    async with aiohttp.ClientSession() as session:
        async with session.get(raw_url) as response:
            if response.status == 200:
                return await response.text()
    return None

def main() -> None:
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("set_strategy", set_strategy))
    application.add_handler(CommandHandler("toggle_ai", toggle_ai))
    application.add_handler(CommandHandler("share_strategy", share_strategy))
    application.add_handler(CommandHandler("import_strategy", import_strategy))
    
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_strategy_input))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_pastebin_link))

    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()