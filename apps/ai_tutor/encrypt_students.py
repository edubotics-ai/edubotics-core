import os
import hashlib
import json
import argparse
from dotenv import load_dotenv


# Function to deterministically hash emails
def deterministic_hash(email, salt):
    return hashlib.pbkdf2_hmac("sha256", email.encode(), salt, 100000).hex()


def main(args):
    # Load the .env file
    load_dotenv()

    # Get the encryption key (salt)
    encryption_salt = os.getenv("EMAIL_ENCRYPTION_KEY").encode()

    # Load emails from the specified JSON file
    with open(args.students_file, "r") as file:
        emails = json.load(file)

    # Replace emails with deterministic hashed emails, {hashed_email: [roles]}
    hashed_emails = {
        deterministic_hash(email, encryption_salt): roles
        for email, roles in emails.items()
    }

    # Save hashed emails to the specified encrypted JSON file
    with open(args.encrypted_students_file, "w") as file:
        json.dump(hashed_emails, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encrypt student emails in a JSON file."
    )
    parser.add_argument(
        "--students-file",
        type=str,
        default="private/students.json",
        help="Path to the students JSON file",
    )
    parser.add_argument(
        "--encrypted-students-file",
        type=str,
        default="public/files/students_encrypted.json",
        help="Path to save the encrypted students JSON file",
    )
    args = parser.parse_args()

    main(args)
