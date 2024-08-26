import os
from dotenv import load_dotenv
import hashlib
import json

# Load the .env file
load_dotenv()

# Get the encryption key (salt)
encryption_salt = os.getenv("EMAIL_ENCRYPTION_KEY").encode()


# Function to deterministically hash emails
def deterministic_hash(email, salt):
    return hashlib.pbkdf2_hmac("sha256", email.encode(), salt, 100000).hex()


# Load emails from private/students.json
with open("private/students.json", "r") as file:
    emails = json.load(file)

# Replace emails with deterministic hashed emails, {hashed_email: [roles]}
hashed_emails = {
    deterministic_hash(email, encryption_salt): roles for email, roles in emails.items()
}

# Save hashed emails to private/students_encrypted.json
with open("private/students_encrypted.json", "w") as file:
    json.dump(hashed_emails, file)
