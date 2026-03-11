import sqlite3
import json
import time
import os

class QueueManager:
    def __init__(self, db_path='data/queue.db'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS data_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    retry_count INTEGER DEFAULT 0,
                    last_attempt REAL
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON data_queue(status)')

    def add(self, data: dict) -> int:
        """Add a data packet to the queue. Returns the row id."""
        payload = json.dumps(data)
        ts = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                'INSERT INTO data_queue (timestamp, payload) VALUES (?, ?)',
                (ts, payload)
            )
            return cur.lastrowid

    def get_pending(self, limit=10) -> list:
        """Retrieve pending records (oldest first)."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute('''
                SELECT id, payload, retry_count FROM data_queue
                WHERE status = 'pending'
                ORDER BY timestamp ASC
                LIMIT ?
            ''', (limit,)).fetchall()
        return [{'id': r[0], 'payload': json.loads(r[1]), 'retry_count': r[2]} for r in rows]

    def mark_sent(self, record_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('UPDATE data_queue SET status = ? WHERE id = ?', ('sent', record_id))

    def mark_failed(self, record_id, increment_retry=True):
        """Mark as failed and optionally increment retry counter."""
        with sqlite3.connect(self.db_path) as conn:
            if increment_retry:
                conn.execute('''
                    UPDATE data_queue
                    SET status = 'pending', retry_count = retry_count + 1, last_attempt = ?
                    WHERE id = ?
                ''', (time.time(), record_id))
            else:
                conn.execute('UPDATE data_queue SET status = ? WHERE id = ?', ('failed', record_id))

    def get_stats(self):
        with sqlite3.connect(self.db_path) as conn:
            stats = conn.execute('''
                SELECT status, COUNT(*) FROM data_queue GROUP BY status
            ''').fetchall()
        return dict(stats)

    def clean_old_sent(self, days=7):
        """Remove 'sent' records older than given days."""
        cutoff = time.time() - days * 86400
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM data_queue WHERE status = ? AND timestamp < ?', ('sent', cutoff))