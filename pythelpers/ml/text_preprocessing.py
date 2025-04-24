import re
import string
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Define stop words for text cleaning
stop_words_engl = set(stopwords.words('english'))
# Initialize lemmatizer for text cleaning
lemmatizer = WordNetLemmatizer()

#Text  Cleaning
def strip_all_entities(body):
    body = re.sub(r'\r|\n', ' ', body.lower())  # Replace newline and carriage return with space, and convert to lowercase
    body = re.sub(r"(?:\@|https?\://)\S+", "", body)  # Remove links
    body = re.sub(r'[^\x00-\x7f]', '', body)  # Remove non-ASCII characters
    banned_list = string.punctuation
    table = str.maketrans('', '', banned_list)
    body = body.translate(table)
    body = ' '.join(word for word in body.split() if word not in stop_words_engl)
    return body

# Filter special characters such as & and $ present in some words
def filter_chars(body):
    return ' '.join('' if ('$' in word) or ('&' in word) else word for word in body.split())

# Remove multiple spaces
def remove_mult_spaces(body):
    return re.sub(r"\s\s+", " ", body)
# Expand contractions
def expand_contractions(body):
    return contractions.fix(body)
# Remove numbers
def remove_numbers(body):
    return re.sub(r'\d+', '', body)
# Lemmatize words
def lemmatize(body):
    words = word_tokenize(body)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)
# Remove short words
def remove_short_words(body, min_len=2):
    words = body.split()
    long_words = [word for word in words if len(word) >= min_len]
    return ' '.join(long_words)
# Replace elongated words with their base form
def replace_elongated_words(body):
    regex_pattern = r'\b(\w+)((\w)\3{2,})(\w*)\b'
    return re.sub(regex_pattern, r'\1\3\4', body)
# Remove repeated punctuation
def remove_repeated_punctuation(body):
    return re.sub(r'[\?\.\!]+(?=[\?\.\!])', '', body)
# Remove extra whitespace
def remove_extra_whitespace(body):
    return ' '.join(body.split())
def remove_url_shorteners(body):
    return re.sub(r'(?:http[s]?://)?(?:www\.)?(?:bit\.ly|goo\.gl|t\.co|tinyurl\.com|tr\.im|is\.gd|cli\.gs|u\.nu|url\.ie|tiny\.cc|alturl\.com|ow\.ly|bit\.do|adoro\.to)\S+', '', body)
# Remove short  tickets
def remove_short_tickets(ticket, min_words=0):    # We do not need this , real data world is not forgiving
    words = ticket.split()
    return ticket if len(words) >= min_words else ""