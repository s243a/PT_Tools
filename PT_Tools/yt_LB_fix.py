import re
import pyperclip
# Get clipboard data
data = str(pyperclip.paste())
print(data)
# Replace the line break with a space using the regex with multiline and non-greedy flags
data = re.sub(r"([0-9]{1,2}):([0-9]{2})..", r"\1:\2 ", data, flags=re.M | re.S | re.U)
print(data)
# Set clipboard data
pyperclip.copy(data)