import argparse
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument('--from_db', default='/root/merge_db/from/.cache.db')
parser.add_argument('--to_db', default='/root/merge_db/to/.cache.db')

args = parser.parse_args()

from_db = sqlite3.connect(args.from_db)
to_db = sqlite3.connect(args.to_db)

for row in from_db.execute("SELECT * FROM downloads"):
    # a,b,c,d,e,f,g,h,i = row
    to_db.execute('INSERT OR REPLACE INTO downloads VALUES (?,?,?,?,?,?,?,?,?)', row)
to_db.commit()