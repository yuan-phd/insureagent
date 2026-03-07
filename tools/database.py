# SQLite Database for the format of client info

import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'insurance.db')

def create_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS policies (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            plan_type TEXT,
            covers TEXT,
            deductible REAL,
            max_annual_payout REAL,
            status TEXT,
            claims_this_year INTEGER
        )
    """)

    policies = [
        ("P-1001", "Alice Murphy",    "Premium",  '["storm","hail","theft","fire","flood","collision"]', 500,  50000, "active",    1),
        ("P-1002", "Brian O'Neill",   "Basic",    '["fire","theft"]',                                    1000, 15000, "active",    0),
        ("P-1003", "Ciara Walsh",     "Standard", '["storm","fire","theft","collision"]',                750,  30000, "active",    3),
        ("P-1004", "David Chen",      "Premium",  '["storm","hail","theft","fire","flood","collision"]', 500,  50000, "lapsed",    0),
        ("P-1005", "Emma Byrne",      "Standard", '["storm","fire","theft","collision"]',                750,  30000, "active",    0),
        ("P-1006", "Fiona Kelly",     "Premium",  '["storm","hail","theft","fire","flood","collision"]', 500,  50000, "active",    0),
        ("P-1007", "Gareth Hughes",   "Basic",    '["fire","theft"]',                                    1000, 15000, "active",    2),
        ("P-1008", "Hannah Doyle",    "Standard", '["storm","fire","theft","collision"]',                750,  30000, "cancelled", 0),
        ("P-1009", "Ivan Petrov",     "Premium",  '["storm","hail","theft","fire","flood","collision"]', 500,  50000, "active",    0),
        ("P-1010", "Julia Brennan",   "Basic",    '["fire","theft"]',                                    1000, 15000, "active",    1),
        ("P-1011", "Kevin McCarthy",  "Standard", '["storm","fire","theft","collision"]',                750,  30000, "active",    4),
        ("P-1012", "Laura Fitzgerald","Premium",  '["storm","hail","theft","fire","flood","collision"]', 500,  50000, "active",    2),
        ("P-1013", "Mark Dunne",      "Basic",    '["fire","theft"]',                                    1000, 15000, "lapsed",    0),
        ("P-1014", "Niamh Power",     "Standard", '["storm","fire","theft","collision"]',                750,  30000, "active",    1),
        ("P-1015", "Oscar Flynn",     "Premium",  '["storm","hail","theft","fire","flood","collision"]', 500,  50000, "active",    9),
        ("P-1016", "Paula Reilly",    "Basic",    '["fire","theft"]',                                    1000, 15000, "active",    0),
        ("P-1017", "Quinn Sheridan",  "Standard", '["storm","fire","theft","collision"]',                750,  30000, "active",    0),
        ("P-1018", "Rachel Nolan",    "Premium",  '["storm","hail","theft","fire","flood","collision"]', 500,  50000, "cancelled", 0),
        ("P-1019", "Sean Gallagher",  "Basic",    '["fire","theft"]',                                    1000, 15000, "active",    0),
        ("P-1020", "Tara Connolly",   "Standard", '["storm","fire","theft","collision"]',                750,  30000, "active",    2),
        ("P-1021", "Ultan Brady",     "Premium",  '["storm","hail","theft","fire","flood","collision"]', 500,  50000, "active",    0),
        ("P-1022", "Vera Costello",   "Basic",    '["fire","theft"]',                                    1000, 15000, "lapsed",    0),
        ("P-1023", "Will Hennessy",   "Standard", '["storm","fire","theft","collision"]',                750,  30000, "active",    3),
        ("P-1024", "Xena Kavanagh",   "Premium",  '["storm","hail","theft","fire","flood","collision"]', 500,  50000, "active",    1),
        ("P-1025", "Yusuf Okonkwo",   "Basic",    '["fire","theft"]',                                    1000, 15000, "active",    0),
        ("P-1026", "Zoe Stafford",    "Standard", '["storm","fire","theft","collision"]',                750,  30000, "active",    0),
        ("P-1027", "Aaron Whelan",    "Premium",  '["storm","hail","theft","fire","flood","collision"]', 500,  50000, "active",    5),
        ("P-1028", "Bella Cronin",    "Basic",    '["fire","theft"]',                                    1000, 15000, "cancelled", 0),
        ("P-1029", "Cormac Daly",     "Standard", '["storm","fire","theft","collision"]',                750,  30000, "active",    1),
        ("P-1030", "Deirdre Smyth",   "Premium",  '["storm","hail","theft","fire","flood","collision"]', 500,  50000, "active",    0),
    ]

    c.executemany("INSERT OR REPLACE INTO policies VALUES (?,?,?,?,?,?,?,?)", policies)
    conn.commit()
    conn.close()
    print(f"Database created with {len(policies)} policies.")

def lookup_policy(user_id: str) -> dict:
    """Look up a policyholder's coverage details by user ID."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM policies WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return {"error": f"No policy found for user_id: {user_id}"}
    return {
        "user_id":            row[0],
        "name":               row[1],
        "plan_type":          row[2],
        "covers":             json.loads(row[3]),
        "deductible":         row[4],
        "max_annual_payout":  row[5],
        "status":             row[6],
        "claims_this_year":   row[7],
    }

if __name__ == "__main__":
    create_database()
    print("Test lookup P-1001:", lookup_policy("P-1001"))
    print("Test lookup P-1004:", lookup_policy("P-1004"))  # lapsed
    print("Test lookup P-9999:", lookup_policy("P-9999"))  # not found