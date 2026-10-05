from contextlib import closing
import sqlite3
import unittest

from scripts.audit_full_organ_progress import prefix_digest, progress_summary


class FullOrganProgressTest(unittest.TestCase):
    def rows(self):
        return [dict(fold=f, role=r, expected=100, stored=0, complete=False, bytes=0)
                for f in range(5) for r in ('fit', 'selection')]

    def test_two_complete_roles_are_not_full_completion(self):
        rows = self.rows()
        for row in rows[:2]:
            row.update(stored=100, complete=True, bytes=2000, added=100, wall_seconds=5.)
        summary = progress_summary(rows, 20*1024**3)
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['completed_roles'], 2)
        self.assertEqual(summary['remaining'], 800)
        self.assertEqual(summary['estimated_remaining_bytes'], 16000)

    def test_missing_roles_and_false_complete_rejected(self):
        with self.assertRaises(ValueError):
            progress_summary(self.rows()[:2], 0)
        rows = self.rows()
        rows[0]['complete'] = True
        with self.assertRaises(ValueError):
            progress_summary(rows, 0)

    def test_all_roles_required_for_completion(self):
        rows = self.rows()
        for row in rows:
            row.update(stored=100, complete=True, bytes=2000)
        summary = progress_summary(rows, 20*1024**3)
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['remaining'], 0)
        self.assertIsNone(summary['estimated_remaining_hours'])

    def test_prefix_unchanged_after_append_but_detects_mutation(self):
        with closing(sqlite3.connect(':memory:')) as db:
            db.execute('CREATE TABLE compartments(patient TEXT,target INTEGER,digest TEXT)')
            db.executemany('INSERT INTO compartments VALUES (?,?,?)', [('a', 0, 'd1'), ('b', 0, 'd2')])
            first = prefix_digest(db, 2)
            db.execute("INSERT INTO compartments VALUES ('c',0,'d3')")
            self.assertEqual(first, prefix_digest(db, 2))
            db.execute("UPDATE compartments SET digest='changed' WHERE patient='a'")
            self.assertNotEqual(first, prefix_digest(db, 2))
            with self.assertRaises(ValueError):
                prefix_digest(db, 4)


if __name__ == '__main__':
    unittest.main()
