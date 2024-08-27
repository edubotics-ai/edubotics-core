# WIP


## Run the encrypt_students script

- If you don't want the emails to be public, run this script to encrypt the emails of the students.
- This will create a new file in the public/files/ directory.
- Place your file with the students' emails in the private/ directory (do not commit this file to the repository).

```bash
python encrypt_students.py --students-file private/students.json --encrypted-students-file public/files/students_encrypted.json
```
