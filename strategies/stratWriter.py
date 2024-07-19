import os

import requests

def upload_to_pastebin(text_string):
  """Uploads a text string to Pastebin and returns the paste link.

  Args:
      text_string (str): The text content to upload.

  Returns:
      str: The URL of the created paste on Pastebin, or None if upload fails.
  """

  # Pastebin API URL (replace with your desired API endpoint if needed)
  url = "https://pastebin.com/api/api_new.php"

  # Payload data with API key (replace with your actual key)
  data = {
      "api_dev_key": "YOUR_PASTEBIN_DEV_KEY",  # Replace with your Pastebin Dev Key
      "api_paste_code": text_string
  }

  # Send POST request
  response = requests.post(url, data=data)

  # Check for successful response
  if response.status_code == 200:
    return response.text  # Pastebin URL is in the response body
  else:
    print(f"Error uploading to Pastebin: {response.status_code}")
    return None

# Example usage
text_to_upload = "This is the text content to be uploaded to Pastebin."
pastebin_url = upload_to_pastebin(text_to_upload)

if pastebin_url:
  print(f"Paste created: {pastebin_url}")
else:
  print("Failed to upload the text.")


def save_user_strategy(strategy):
  """Saves the user's strategy to a file named after the keyword.

  Args:
      strategy (str): The user's strategy content.

  Returns:
      str: The path to the saved strategy file.
  """
  #strategy= input('input your strategy : ')

  # Extract keyword (assuming first word)
  keyword = strategy.split()[0].upper()

  # Create filename with .strat extension
  filename = f"{keyword}.strat"

  # Get full path, ensuring directory exists
  filepath = os.path.join(os.getcwd(), filename)
  os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Create directory if needed


#upload to pastebin
#strat_url = pastebin.upload(strategy)
#strategy = strat_url + "\n" + strategy

  # Save strategy to file
  with open(filepath, 'w') as strat_file:
    strat_file.write(strategy)

  return filepath

# Example usage (assuming you have a strategy variable)
user_strategy = "RSI strategy for stock trading..."
saved_path = save_user_strategy(user_strategy)
print(f"Strategy saved to: {saved_path}")
