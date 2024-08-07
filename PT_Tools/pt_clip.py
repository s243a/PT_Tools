# Import the modules
import os # for creating files
import re # for using regular expressions
import datetime # for working with dates and times
from bs4 import BeautifulSoup # for parsing HTML
#import win32clipboard # for accessing the clipboard
from jaraco import clipboard #As jaraco.clipboard
#import win32com.client
# Open the clipboard
#win32clipboard.OpenClipboard()

# Get the clipboard content as a string in HTML format
# The format ID for HTML is 49424
#content = win32clipboard.GetClipboardData(49424)
#content=jaraco.clipboard.paste_html()
content=clipboard.paste_html()
# Close the clipboard
#win32clipboard.CloseClipboard()

# Parse the content as HTML using BeautifulSoup
soup = BeautifulSoup(content, "html.parser")

# Print the content to the terminal with HTML tags
print(soup.prettify())

# Find all the links in the HTML using the <a> tag
links = soup.find_all("a")

# Get the current date and time as a string
# The format is YYYY-MM-DD_HH-MM-SS
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create a subdirectory with the date and time as the name
#os.mkdir(now)
the_path_str=r'C:\Users\johnc\Favorites' + "\\" + now
print("the_path_str="+the_path_str)

the_path=os.path.abspath(the_path_str)
print()
os.mkdir(the_path)
#input("Press Enter to continue...")       
# Loop through each link
for link in links:
    # Get the link URL from the href attribute
    url = link["href"]
    title=link.text
    print("url1="+url)
    print("url1.text="+link.text)
    # Check if the URL is valid
    if url.startswith("http://") or url.startswith("https://"):
        # Get the file name from the URL by removing the protocol and slashes
        print("url2="+url)
        file_name = (title.replace("http://", "")
                          .replace("https://", "")
                          .replace(":","%3A")
                          .replace("/", "%2F")
                          .replace("*", "_star_")
                          .replace('"',"%22")
                          .replace(':',"%3A")
                          .replace('?',"%3F"))
        # Add the .url extension
        file_name = file_name + ".url"
        # Join the subdirectory name and the file name


        file_path = os.path.join(the_path, file_name)
        # Create a new file with the file path
        file = open(file_path, "w")
        # Write the .url file header
        file.write("[InternetShortcut]\n")
        # Write the URL
        file.write(f"URL={url}\n")
        # Close the file
        file.close()
