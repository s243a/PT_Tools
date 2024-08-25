import abc
import hashlib
import sqlite3
import lxml.etree as ET
from bs4 import BeautifulSoup
from jaraco import clipboard
from crate import client
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Number of digits to use from the MD5 hash for the default folder name
DEFAULT_HASH_DIGITS = 5
# Prefix for fallback folder names
FALLBACK_PREFIX = "md5_"

# ... [Database classes remain the same] ...

class AbstractDatabase(abc.ABC):
    @abc.abstractmethod
    def connect(self):
        pass

    @abc.abstractmethod
    def create_table(self):
        pass

    @abc.abstractmethod
    def insert_tree(self, tree_id, title):
        pass

    @abc.abstractmethod
    def get_tree_id(self, title):
        pass

    @abc.abstractmethod
    def close(self):
        pass

class CrateDatabase(AbstractDatabase):
    def __init__(self, host="http://localhost:4200"):
        self.host = host
        self.connection = None

    def connect(self):
        self.connection = client.connect(self.host)
        return self.connection.cursor()

    def create_table(self):
        cursor = self.connect()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trees (
                tree_id TEXT PRIMARY KEY,
                title TEXT INDEX USING FULLTEXT
            )
        """)

    def insert_tree(self, tree_id, title):
        cursor = self.connect()
        cursor.execute(
            "INSERT INTO trees (tree_id, title) VALUES (?, ?) ON CONFLICT (tree_id) DO UPDATE SET title = ?",
            (tree_id, title, title)
        )

    def get_tree_id(self, title):
        cursor = self.connect()
        cursor.execute("""
            SELECT tree_id, title
            FROM trees
            WHERE match(title, ?)
            ORDER BY _score DESC
            LIMIT 1
        """, (title,))
        result = cursor.fetchone()
        return result[0] if result else None

    def close(self):
        if self.connection:
            self.connection.close()

class SQLiteDatabase(AbstractDatabase):
    def __init__(self, db_name="pearltrees.db"):
        self.db_name = db_name
        self.connection = None

    def connect(self):
        self.connection = sqlite3.connect(self.db_name)
        return self.connection.cursor()

    def create_table(self):
        cursor = self.connect()
        cursor.execute('''CREATE TABLE IF NOT EXISTS trees
                         (tree_id TEXT PRIMARY KEY, title TEXT)''')

    def insert_tree(self, tree_id, title):
        cursor = self.connect()
        cursor.execute("INSERT OR REPLACE INTO trees VALUES (?, ?)", (tree_id, title))

    def get_tree_id(self, title):
        cursor = self.connect()
        cursor.execute("SELECT tree_id FROM trees WHERE title = ?", (title,))
        result = cursor.fetchone()
        return result[0] if result else None

    def close(self):
        if self.connection:
            self.connection.commit()
            self.connection.close()

def parse_rdf(rdf_file, db):
    tree = ET.parse(rdf_file)
    root = tree.getroot()
    
    db.create_table()

    for tree in root.findall('.//{http://www.pearltrees.com/rdf/0.1/#}Tree'):
        tree_id = tree.find('{http://www.pearltrees.com/rdf/0.1/#}treeId').text
        title = tree.find('{http://purl.org/dc/elements/1.1/}title').text
        db.insert_tree(tree_id, title)

def get_default_folder_name(folder_name, hash_digits=DEFAULT_HASH_DIGITS):
    """Generate a default folder name using the first N digits of the MD5 hash of the folder name."""
    md5_hash = hashlib.md5(folder_name.encode()).hexdigest()
    return f"{FALLBACK_PREFIX}{md5_hash[:hash_digits]}"

def process_bookmark_folder(folder_header, db, parent_file='', hash_digits=DEFAULT_HASH_DIGITS, root_name="index.html"):

    if folder_header is None:
        logging.warning(f"No header found for folder. Using default name.")
        folder_name = "Unnamed Folder"
    else:
        folder_name = folder_header.text
   
    if len(parent_file)>0:


        tree_id = db.get_tree_id(folder_name)
     
     
        if tree_id is None:
            tree_id = get_default_folder_name(folder_name, hash_digits)
            logging.info(f"Using fallback name for '{folder_name}': {tree_id}")
        else:
            logging.info(f"Found tree_id for '{folder_name}': {tree_id}")

        file_name = f"{tree_id}.html"
        logging.info(f"Creating file: {file_name}")
    else:
        file_name=root_name
    logging.info(f"Creating file: {file_name}")

    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(f"<html><head><title>{folder_name}</title></head><body>")
        f.write(f"<h1>{folder_name}</h1>")
        if file_name != root_name:  # Don't add 'Back to parent' link for the root folder
            f.write(f'<p><a href="{parent_file}">Back to parent</a></p>')
        
        f.write("<ul>")

        # Process immediate children (links, subfolders, and section headers)
        folder = folder_header.find_next(['dl', 'DL'])
        for item in folder.children:
            if item.name and item.name.lower() == 'dt':
                dt_content = item.find(['a', 'A', 'h3', 'H3'])
                if dt_content:
                    if dt_content.name.lower() == 'a':
                        # This is a link
                        f.write(f'<li><a href="{dt_content["href"]}">{dt_content.text}</a></li>')
                    elif dt_content.name.lower() == 'h3':
                        # This is a subfolder
                        subfolder_name = dt_content.text
                        subfolder_tree_id = db.get_tree_id(subfolder_name)
                        if subfolder_tree_id is None:
                            subfolder_tree_id = get_default_folder_name(subfolder_name, hash_digits)
                            logging.info(f"Using fallback name for subfolder '{subfolder_name}': {subfolder_tree_id}")
                        else:
                            logging.info(f"Found tree_id for subfolder '{subfolder_name}': {subfolder_tree_id}")
                        subfolder_file = f"{subfolder_tree_id}.html"
                        f.write(f'<li><a href="{subfolder_file}">{subfolder_name}</a></li>')
                        
                        # Process the subfolder
                        next_dl = item.find_next(['dl', 'DL'])
                        if next_dl:
                            process_bookmark_folder(dt_content, db, file_name, hash_digits)
                        else:
                            logging.warning(f"No content found for subfolder '{subfolder_name}'")
                else:
                    logging.warning(f"Found a DT tag without expected content. Skipping this item.")
            elif item.name and item.name.lower() == 'hr':
                # This is a section divider
                f.write('<hr>')
            elif item.name and item.name.lower() == 'dd':
                # Extract the text following the DD tag
                dd_text = item.next_sibling
                if dd_text and isinstance(dd_text, str):
                    f.write(f'<h4>{dd_text.strip()}</h4>')
                else:
                    logging.warning(f"Found a DD tag without expected content. Skipping this item.")
            elif isinstance(item, str) and item.strip():
                # This is likely a section header
                f.write(f'<h2>{item.strip()}</h2>')

        f.write("</ul></body></html>")

    return file_name

def main():
    # Uncomment the database you want to use
    # db = CrateDatabase()
    db = SQLiteDatabase()

    rdf_file = 'Example_pearltrees_rdf_export.rdf'
    logging.info(f"Parsing RDF file: {rdf_file}")
    parse_rdf(rdf_file, db)

    logging.info("Reading clipboard content")
    clipboard_content = clipboard.paste_html()
    soup = BeautifulSoup(clipboard_content, 'html.parser')

    root_folder = soup.find(['H3', 'h3'])
    if root_folder:
        logging.info("Found root folder, starting processing")
        root_file = process_bookmark_folder(root_folder, db, root_name='index.html')
        # Rename the root file to index.html
        #os.rename(root_file, 'index.html')
        #logging.info(f"Renamed root file to index.html")
    else:
        logging.error("No bookmark folder found in clipboard content.")

    db.close()
    logging.info("Processing complete")

if __name__ == "__main__":
    main()