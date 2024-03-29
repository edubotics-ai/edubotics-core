# modules/firebase_admin_init.py
import firebase_admin
from firebase_admin import credentials

cred = credentials.Certificate('famous-tree-389606-firebase-adminsdk-iljhx-0358d21dde.json')
firebase_admin.initialize_app(cred)
