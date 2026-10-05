import unittest
from scripts.partition_additional_domains import partition, ROLES


class PartitionTests(unittest.TestCase):
    def population(self):
        sites={f'p{i:02d}_{j}':{f's{i:02d}'} for i in range(30) for j in range(i%5+2)}
        sites['bridge']={'s00','s01'}
        return sites

    def test_complete_disjoint_roles_and_outer_once(self):
        sites=self.population();folds=partition(sites)
        for f in folds:
            self.assertEqual(set(f),set(sites));self.assertEqual(set(f.values()),set(ROLES))
            for site in set.union(*sites.values()):
                self.assertEqual(len({f[p] for p,s in sites.items() if site in s}),1)
        for p in sites:self.assertEqual(sum(f[p]=='test' for f in folds),1)

    def test_input_order_invariance(self):
        sites=self.population()
        self.assertEqual(partition(sites),partition(dict(reversed(list(sites.items())))))

    def test_too_few_components_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'twelve'):
            partition({'p':{'s'}})
