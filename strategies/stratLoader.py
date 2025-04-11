import requests  # Assuming you don't have a pastebin library

def download_and_save_strategy(strat_link):
  """Downloads a strategy from a pastebin link and saves it as a .strat file.

  Args:
      strat_link (str): The URL of the pastebin containing the strategy.

  Raises:
      ValueError: If the link is invalid'.
      ConnectionError: If there's a network issue while downloading.
      IOError: If there's an error saving the file.
  """

  # Validate file extension
  if "pastebin" not in strat_link:
    raise ValueError("Invalid link: Must point to a strategy file")

  try:
    # Download strategy content
    response = requests.get(strat_link)
    # Raise an exception for non-200 status codes
    #response.raise_for_status()  
    strategy_content = response.text

    # Extract filename from strategy (assuming basic format)
    filename = strategy_content[115:121] + ".strat"

    # Save strategy to file
    with open(filename, 'w') as strat_file:
      strat_file.write(strategy_content)

    print(f"Strategy saved successfully: {filename}")
  except (ValueError, ConnectionError, IOError) as e:
    print(f"Error downloading/saving strategy: {e}")

# Example usage
strat_link = input("Enter strategy link (points to pastebin): ")
download_and_save_strategy(strat_link)


#https://free.facebook.com/pastebin