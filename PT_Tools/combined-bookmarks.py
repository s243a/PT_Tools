#!/usr/bin/env python3
"""
Combined Bookmark Handler
------------------------
This script combines the functionality of pt_clip.py and export_favs.py into a single workflow.
It can scrape URLs from a website, categorize them according to customizable heading mappings,
and export them to Netscape bookmark format for import into Pearltrees or other bookmarking services.

Features:
- Cross-platform support (Windows, Linux, Android/Termux)
- Custom heading mappings for different section types
- Direct URL scraping from a provided website
- Export to Netscape bookmark format
"""

import os
import re
import sys
import datetime
import argparse
import platform
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
from pathlib import Path
import html

# Constants and defaults
DEFAULT_HEADING_MAPPINGS = {
    "See also": "See also",
    "Categories": "Super Categories",
    "Subcategories": "Subcategories",
    "Pages": "Subtopics"
}

# Platform detection
IS_WINDOWS = platform.system() == "Windows"
IS_ANDROID = False

try:
    if "ANDROID_ROOT" in os.environ:
        IS_ANDROID = True
except:
    pass

# Default directories based on platform
if IS_WINDOWS:
    try:
        username = os.environ.get("USERNAME")
        DEFAULT_FAVORITES_PATH = os.path.join("C:\\Users", username, "Favorites")
        DEFAULT_OUTPUT_DIR = os.path.join("C:\\Users", username, "Documents")
    except:
        DEFAULT_FAVORITES_PATH = os.path.join(os.path.expanduser("~"), "Favorites")
        DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Documents")
elif IS_ANDROID:
    DEFAULT_FAVORITES_PATH = os.path.join("/data/data/com.termux/files/home", "favorites")
    DEFAULT_OUTPUT_DIR = os.path.join("/data/data/com.termux/files/home", "documents")
else:  # Linux/macOS
    DEFAULT_FAVORITES_PATH = os.path.join(os.path.expanduser("~"), "favorites")
    DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Documents")


def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(DEFAULT_FAVORITES_PATH, exist_ok=True)
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    return DEFAULT_FAVORITES_PATH, DEFAULT_OUTPUT_DIR


def sanitize_filename(title):
    """
    Sanitize a title to be used as a filename, following the same encoding as pt_clip.py.
    Preserves Unicode characters while handling invalid filename characters.
    """
    # Replace characters that are invalid in filenames
    return (title.replace("http://", "")
                .replace("https://", "")
                .replace(":", "%3A")
                .replace("/", "%2F")
                .replace("*", "_star_")
                .replace('"', "%22")
                .replace('?', "%3F")
                .replace('<', "%3C")
                .replace('>', "%3E")
                .replace('|', "%7C")
                .replace('\\', "%5C"))
                # Not encoding other Unicode characters to preserve them


def decode_filename(encoded_title):
    """
    Decode URL-encoded characters in a title based on the specific encoding used.
    """
    # Remove the .url extension if present
    if encoded_title.lower().endswith('.url'):
        encoded_title = encoded_title[:-4]
    
    # Reverse the specific replacements that were made during encoding
    decoded = (encoded_title.replace("%3A", ":")
                          .replace("%2F", "/")
                          .replace("_star_", "*")
                          .replace("%2A", "*")
                          .replace("%22", '"')
                          .replace("%3F", "?"))
    
    return decoded


def scrape_links_from_url(url, heading_mappings=None):
    """
    Scrape links from a given URL, organizing them based on section headings.
    Returns a dictionary of sections and their links.
    """
    if heading_mappings is None:
        heading_mappings = DEFAULT_HEADING_MAPPINGS

    sections = {}
    
    try:
        # Fetch the webpage content
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            html_content = response.read().decode('utf-8', errors='replace')
            
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Extract page title for the main folder name
        page_title = soup.title.text if soup.title else "Scraped Links"
        
        # Initialize with main page info
        sections["main"] = {
            "title": page_title,
            "links": [{"url": url, "title": page_title}]
        }
        
        # Process Wikipedia-specific sections if the URL is from Wikipedia
        if "wikipedia.org" in url:
            # Look for "See also" section
            see_also_heading = soup.find(id="See_also") or soup.find(string="See also")
            if see_also_heading:
                see_also_section = see_also_heading.find_parent("h2")
                if see_also_section:
                    see_also_links = []
                    current = see_also_section.next_sibling
                    while current and not (current.name == "h2"):
                        if current.name == "ul":
                            for li in current.find_all("li"):
                                a_tag = li.find("a")
                                if a_tag and a_tag.has_attr("href") and not a_tag.has_attr("class"):
                                    href = a_tag["href"]
                                    if href.startswith("/wiki/") and ":" not in href:
                                        full_url = "https://en.wikipedia.org" + href
                                        see_also_links.append({
                                            "url": full_url,
                                            "title": a_tag.text
                                        })
                        current = current.next_sibling
                    
                    if see_also_links:
                        sections[heading_mappings["See also"]] = {
                            "title": heading_mappings["See also"],
                            "links": see_also_links
                        }
            
            # Process categories
            categories_div = soup.find(id="mw-normal-catlinks")
            if categories_div:
                category_links = []
                for a_tag in categories_div.find_all("a"):
                    if a_tag.has_attr("href") and a_tag["href"].startswith("/wiki/Category:"):
                        full_url = "https://en.wikipedia.org" + a_tag["href"]
                        category_links.append({
                            "url": full_url,
                            "title": a_tag.text
                        })
                
                if category_links:
                    sections[heading_mappings["Categories"]] = {
                        "title": heading_mappings["Categories"],
                        "links": category_links
                    }
            
            # Look for subcategories on category pages
            if "/wiki/Category:" in url:
                subcats_div = soup.find(id="mw-subcategories")
                if subcats_div:
                    subcat_links = []
                    for a_tag in subcats_div.find_all("a"):
                        if a_tag.has_attr("href") and a_tag["href"].startswith("/wiki/Category:"):
                            full_url = "https://en.wikipedia.org" + a_tag["href"]
                            subcat_links.append({
                                "url": full_url,
                                "title": a_tag.text
                            })
                    
                    if subcat_links:
                        sections[heading_mappings["Subcategories"]] = {
                            "title": heading_mappings["Subcategories"],
                            "links": subcat_links
                        }
                
                # Look for pages in the category
                pages_div = soup.find(id="mw-pages")
                if pages_div:
                    page_links = []
                    for a_tag in pages_div.find_all("a"):
                        if a_tag.has_attr("href") and a_tag["href"].startswith("/wiki/") and ":" not in a_tag["href"]:
                            full_url = "https://en.wikipedia.org" + a_tag["href"]
                            page_links.append({
                                "url": full_url,
                                "title": a_tag.text
                            })
                    
                    if page_links:
                        sections[heading_mappings["Pages"]] = {
                            "title": heading_mappings["Pages"],
                            "links": page_links
                        }
        
        # For non-Wikipedia pages, just collect all links
        else:
            all_links = []
            for a_tag in soup.find_all("a"):
                if a_tag.has_attr("href"):
                    href = a_tag["href"]
                    
                    # Convert relative URLs to absolute
                    if href.startswith("http") or href.startswith("https"):
                        full_url = href
                    elif href.startswith("//"):
                        full_url = "https:" + href
                    elif href.startswith("/"):
                        base_url = urllib.parse.urlparse(url)
                        full_url = f"{base_url.scheme}://{base_url.netloc}{href}"
                    else:
                        # Skip javascript and other non-HTTP URLs
                        if href.startswith("javascript:") or href.startswith("#"):
                            continue
                        # Handle relative URLs properly
                        base_url = url
                        if not base_url.endswith("/"):
                            base_url = base_url.rsplit("/", 1)[0] + "/"
                        full_url = urllib.parse.urljoin(base_url, href)
                    
                    # Use text as title, with fallback to URL
                    title = a_tag.text.strip() or href
                    
                    all_links.append({
                        "url": full_url,
                        "title": title
                    })
            
            if all_links:
                sections["Links"] = {
                    "title": "Links",
                    "links": all_links
                }
        
        return sections
    
    except Exception as e:
        print(f"Error scraping links from {url}: {e}")
        return {"main": {"title": "Error", "links": []}}


def split_links_alphabetically(links, max_links_per_group=35):
    """
    Split a list of links into alphabetical groups, trying to keep groups
    between 25-50 items or as evenly distributed as possible.
    
    Returns a list of tuples (range_name, links_in_range)
    """
    if not links:
        return []
    
    # Sort links alphabetically by title
    sorted_links = sorted(links, key=lambda x: x["title"].lower())
    
    # If fewer than max_links_per_group, return as a single group
    if len(sorted_links) <= max_links_per_group:
        return [("", sorted_links)]
    
    # Calculate ideal number of groups
    ideal_group_count = max(2, len(sorted_links) // max_links_per_group)
    
    # Get first letter of each title
    first_letters = [link["title"][0].upper() if link["title"] else "#" for link in sorted_links]
    unique_letters = sorted(set(first_letters))
    
    # If fewer unique letters than ideal groups, use the unique letters
    if len(unique_letters) <= ideal_group_count:
        groups = []
        current_group = []
        current_letter = None
        
        for i, link in enumerate(sorted_links):
            first_letter = first_letters[i]
            
            if first_letter != current_letter:
                if current_group:
                    # Determine range name
                    if len(groups) == 0:
                        range_name = f"{current_letter}"
                    else:
                        prev_group_letter = groups[-1][0]
                        range_name = f"{prev_group_letter}-{current_letter}"
                    
                    groups.append((range_name, current_group))
                    current_group = []
                current_letter = first_letter
            
            current_group.append(link)
        
        # Add the last group
        if current_group:
            if len(groups) == 0:
                range_name = f"{current_letter}"
            else:
                prev_group_letter = groups[-1][0]
                range_name = f"{prev_group_letter}-{current_letter}"
            
            groups.append((range_name, current_group))
        
        # Merge small groups if needed
        if len(groups) > 1:
            i = 0
            while i < len(groups) - 1:
                if len(groups[i][1]) < 15 and len(groups[i][1]) + len(groups[i+1][1]) <= max_links_per_group * 1.5:
                    # Merge groups
                    merged_links = groups[i][1] + groups[i+1][1]
                    start_letter = groups[i][0].split("-")[0]
                    end_letter = groups[i+1][0].split("-")[-1]
                    merged_name = f"{start_letter}-{end_letter}"
                    groups[i] = (merged_name, merged_links)
                    groups.pop(i+1)
                else:
                    i += 1
        
        return groups
    
    # Otherwise, divide links evenly into the ideal number of groups
    else:
        links_per_group = len(sorted_links) // ideal_group_count
        groups = []
        
        for i in range(ideal_group_count):
            start_idx = i * links_per_group
            end_idx = (i + 1) * links_per_group if i < ideal_group_count - 1 else len(sorted_links)
            
            group_links = sorted_links[start_idx:end_idx]
            
            if not group_links:
                continue
                
            # Determine range name (first letter of first and last link in group)
            first_link_letter = group_links[0]["title"][0].upper() if group_links[0]["title"] else "#"
            last_link_letter = group_links[-1]["title"][0].upper() if group_links[-1]["title"] else "#"
            
            if first_link_letter == last_link_letter:
                range_name = f"{first_link_letter}"
            else:
                range_name = f"{first_link_letter}-{last_link_letter}"
            
            groups.append((range_name, group_links))
        
        return groups


def save_to_url_files(sections, favorites_path):
    """
    Save the scraped sections and links to .url files in the favorites directory.
    Returns a dictionary mapping section names to their directories.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    main_dir = os.path.join(favorites_path, timestamp)
    os.makedirs(main_dir, exist_ok=True)
    
    section_directories = {}
    
    # Handle the main section (page itself)
    if "main" in sections:
        main_info = sections["main"]
        main_title = main_info["title"]
        
        # Create .url file for the main page
        for link in main_info["links"]:
            file_name = sanitize_filename(link["title"]) + ".url"
            file_path = os.path.join(main_dir, file_name)
            
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("[InternetShortcut]\n")
                    f.write(f"URL={link['url']}\n")
                    f.write(f"TITLE={link['title']}\n")
            except UnicodeEncodeError:
                # If UTF-8 encoding fails, try with a different encoding that can handle all characters
                with open(file_path, "w", encoding="utf-8-sig") as f:
                    f.write("[InternetShortcut]\n")
                    f.write(f"URL={link['url']}\n")
                    f.write(f"TITLE={link['title']}\n")
    
    # Process each section
    for section_name, section_data in sections.items():
        if section_name == "main":
            continue
        
        links = section_data["links"]
        section_title = section_data["title"]
        
        # If section has more than 35 links, split it alphabetically
        if len(links) > 35:
            alphabetical_groups = split_links_alphabetically(links)
            
            for group_range, group_links in alphabetical_groups:
                # Create subdirectory name based on the alphabetical range
                if group_range:
                    subsection_title = f"{section_title} ({group_range})"
                else:
                    subsection_title = section_title
                
                # Create a subdirectory for this alphabetical group
                section_dir = os.path.join(main_dir, subsection_title)
                os.makedirs(section_dir, exist_ok=True)
                section_directories.setdefault(section_name, []).append(section_dir)
                
                # Save each link in this group
                for link in group_links:
                    file_name = sanitize_filename(link["title"]) + ".url"
                    file_path = os.path.join(section_dir, file_name)
                    
                    try:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write("[InternetShortcut]\n")
                            f.write(f"URL={link['url']}\n")
                            f.write(f"TITLE={link['title']}\n")
                    except UnicodeEncodeError:
                        # If UTF-8 encoding fails, try with a different encoding that can handle all characters
                        with open(file_path, "w", encoding="utf-8-sig") as f:
                            f.write("[InternetShortcut]\n")
                            f.write(f"URL={link['url']}\n")
                            f.write(f"TITLE={link['title']}\n")
        else:
            # Create a subdirectory for this section
            section_dir = os.path.join(main_dir, section_data["title"])
            os.makedirs(section_dir, exist_ok=True)
            section_directories[section_name] = section_dir
            
            # Save each link in this section
            for link in section_data["links"]:
                file_name = sanitize_filename(link["title"]) + ".url"
                file_path = os.path.join(section_dir, file_name)
                
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write("[InternetShortcut]\n")
                        f.write(f"URL={link['url']}\n")
                        f.write(f"TITLE={link['title']}\n")
                except UnicodeEncodeError:
                    # If UTF-8 encoding fails, try with a different encoding that can handle all characters
                    with open(file_path, "w", encoding="utf-8-sig") as f:
                        f.write("[InternetShortcut]\n")
                        f.write(f"URL={link['url']}\n")
                        f.write(f"TITLE={link['title']}\n")
    
    return main_dir, section_directories


def escape_html_content(text):
    """
    Escape special characters for HTML content while preserving existing entities.
    """
    # Replace ampersands that aren't part of valid HTML entities
    text = re.sub(r'&(?!(#[0-9]{1,7};|#x[0-9a-fA-F]{1,6};|[a-zA-Z]{1,8};))', '&amp;', text)
    
    # Escape < and > normally
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    
    return text


def read_url_file_data(file_path):
    """
    Read a .url file and extract both the URL and title if available.
    """
    url = None
    title = None
    
    # Try multiple encodings in order of preference
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("URL="):
                        url = line[4:]
                    elif line.startswith("TITLE=") or line.startswith("Title="):
                        title = line.split('=', 1)[1]
            # If we got here without exception, break the loop
            if url or title:
                break
        except UnicodeDecodeError:
            # Try the next encoding
            continue
        except Exception as e:
            print(f"Error reading file {file_path} with {encoding} encoding: {e}")
            break
    
    return url, title


def export_to_netscape_format(directory, output_dir):
    """
    Export the .url files in the given directory to Netscape bookmark format.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_dir, f"bookmarks-{timestamp}.html")
    
    with open(output_file, "w", encoding="utf-8-sig") as f:
        # Write header
        f.write("<!DOCTYPE NETSCAPE-Bookmark-file-1>\n")
        f.write("<META HTTP-EQUIV=\"Content-Type\" CONTENT=\"text/html; charset=UTF-8\">\n")
        f.write("<TITLE>Bookmarks</TITLE>\n")
        f.write("<H1>Bookmarks</H1>\n")
        f.write("<DL><p>\n")
        
        # Get the name of the main directory (timestamp)
        main_dir_name = os.path.basename(directory)
        f.write(f'  <DT><H3 FOLDED ADD_DATE="{int(datetime.datetime.now().timestamp())}">{main_dir_name}</H3>\n')
        f.write("  <DL><p>\n")
        
        # Process files directly in the main directory first
        for item in os.scandir(directory):
            if item.is_file() and item.name.lower().endswith('.url'):
                url, file_title = read_url_file_data(item.path)
                
                if url is None:
                    continue
                
                # Get title from file or filename
                if file_title:
                    title = file_title
                else:
                    raw_title = item.name[:-4]  # Remove .url extension
                    title = decode_filename(raw_title)
                
                title = escape_html_content(title)
                safe_url = url.replace('"', "&quot;")
                
                f.write(f'    <DT><A HREF="{safe_url}" ADD_DATE="{int(datetime.datetime.now().timestamp())}">{title}</A>\n')
        
        # Process subdirectories - organize by section groups
        # First, collect all subdirectories and group by section type
        section_groups = {}
        for item in os.scandir(directory):
            if item.is_dir():
                folder_name = item.name
                # Extract section name without the alphabetical range
                if "(" in folder_name and ")" in folder_name:
                    section_base = folder_name.split("(")[0].strip()
                    if section_base not in section_groups:
                        section_groups[section_base] = []
                    section_groups[section_base].append(item)
                else:
                    if folder_name not in section_groups:
                        section_groups[folder_name] = []
                    section_groups[folder_name].append(item)
        
        # Now process each section group
        for section_name, section_items in section_groups.items():
            # If there's only one directory in this section, don't create an extra level
            if len(section_items) == 1 and "(" not in section_items[0].name:
                item = section_items[0]
                folder_name = item.name
                f.write(f'    <DT><H3 FOLDED ADD_DATE="{int(datetime.datetime.now().timestamp())}">{folder_name}</H3>\n')
                f.write("    <DL><p>\n")
                
                # Process files in this subdirectory
                for subitem in os.scandir(item.path):
                    if subitem.is_file() and subitem.name.lower().endswith('.url'):
                        url, file_title = read_url_file_data(subitem.path)
                        
                        if url is None:
                            continue
                        
                        # Get title from file or filename
                        if file_title:
                            title = file_title
                        else:
                            raw_title = subitem.name[:-4]  # Remove .url extension
                            title = decode_filename(raw_title)
                        
                        title = escape_html_content(title)
                        safe_url = url.replace('"', "&quot;")
                        
                        f.write(f'      <DT><A HREF="{safe_url}" ADD_DATE="{int(datetime.datetime.now().timestamp())}">{title}</A>\n')
                
                f.write("    </DL><p>\n")
            else:
                # For sections with multiple alphabetical ranges, create a parent folder
                f.write(f'    <DT><H3 FOLDED ADD_DATE="{int(datetime.datetime.now().timestamp())}">{section_name}</H3>\n')
                f.write("    <DL><p>\n")
                
                # Sort the subdirectories by their alphabetical range
                sorted_items = sorted(section_items, key=lambda x: x.name)
                
                for item in sorted_items:
                    folder_name = item.name
                    # If the folder has an alphabetical range, extract just the range part
                    if "(" in folder_name and ")" in folder_name:
                        range_part = folder_name[folder_name.index("("):].strip()
                        f.write(f'      <DT><H3 FOLDED ADD_DATE="{int(datetime.datetime.now().timestamp())}">{range_part}</H3>\n')
                    else:
                        # Shouldn't get here for multiple directories of the same section type
                        # but just in case, use the full folder name
                        f.write(f'      <DT><H3 FOLDED ADD_DATE="{int(datetime.datetime.now().timestamp())}">{folder_name}</H3>\n')
                    
                    f.write("      <DL><p>\n")
                    
                    # Process files in this subdirectory
                    for subitem in os.scandir(item.path):
                        if subitem.is_file() and subitem.name.lower().endswith('.url'):
                            url, file_title = read_url_file_data(subitem.path)
                            
                            if url is None:
                                continue
                            
                            # Get title from file or filename
                            if file_title:
                                title = file_title
                            else:
                                raw_title = subitem.name[:-4]  # Remove .url extension
                                title = decode_filename(raw_title)
                            
                            title = escape_html_content(title)
                            safe_url = url.replace('"', "&quot;")
                            
                            f.write(f'        <DT><A HREF="{safe_url}" ADD_DATE="{int(datetime.datetime.now().timestamp())}">{title}</A>\n')
                    
                    f.write("      </DL><p>\n")
                
                f.write("    </DL><p>\n")
        
        f.write("  </DL><p>\n")
        f.write("</DL><p>\n")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Scrape links from a URL and export to Netscape bookmark format")
    parser.add_argument("url", help="URL to scrape links from")
    parser.add_argument("--favorites-dir", help="Directory to store .url files", default=DEFAULT_FAVORITES_PATH)
    parser.add_argument("--output-dir", help="Directory to save output bookmark file", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mapping", help="Custom heading mapping in format 'original:mapped,original2:mapped2'")
    args = parser.parse_args()
    
    # Set up directories
    favorites_path = args.favorites_dir
    output_dir = args.output_dir
    os.makedirs(favorites_path, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process custom heading mappings if provided
    heading_mappings = DEFAULT_HEADING_MAPPINGS.copy()
    if args.mapping:
        try:
            mappings = args.mapping.split(",")
            for mapping in mappings:
                original, mapped = mapping.split(":")
                heading_mappings[original.strip()] = mapped.strip()
        except:
            print("Warning: Could not parse custom mapping. Using defaults.")
    
    print(f"Scraping links from {args.url}...")
    sections = scrape_links_from_url(args.url, heading_mappings)
    
    print(f"Saving links to .url files in {favorites_path}...")
    main_dir, section_dirs = save_to_url_files(sections, favorites_path)
    
    print(f"Exporting to Netscape bookmark format...")
    output_file = export_to_netscape_format(main_dir, output_dir)
    
    print(f"Done! Bookmarks exported to {output_file}")
    print(f"Import this file into Pearltrees or your preferred bookmarking service.")


if __name__ == "__main__":
    main()