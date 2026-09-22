import runpy
import json
import unittest
from pathlib import Path
m = runpy.run_path(str(Path(__file__).with_name('find-gpus')))
def sample(states, metrics=None, processes=''):
    metrics = metrics or '\n'.join(f'{i}, GPU-{i}, H100, 0, 0' for i in range(len(states)))
    parts = [('STATUS', json.dumps([{'gpu_id':i, 'status':s} for i,s in enumerate(states)])), ('METRICS',metrics), ('PROCESSES',processes)]
    return ''.join(f'__GPU_FREE_{k}__\n{v}\n__GPU_FREE_{k}_RC__0\n' for k,v in parts)
class TestAvailability(unittest.TestCase):
    def test_reserved_idle_and_unreserved_busy_are_not_free(self):
        r=m['classify'](sample(['AVAILABLE','IN_USE','UNRESERVED']),256)
        self.assertEqual(r['free'],[0]); self.assertEqual(r['reserved'],[1]); self.assertEqual(r['unreserved_busy'],[2])
    def test_usage_overrides_available(self):
        r=m['classify'](sample(['AVAILABLE']*4,'0, GPU-0, H100, 0, 0\n1, GPU-1, H100, 1, 0\n2, GPU-2, H100, 0, 257\n3, GPU-3, H100, 0, 0','GPU-0, 123'),256)
        self.assertEqual(r['free'],[3]); self.assertEqual(r['unreserved_busy'],[0,1,2])
    def test_unknown_telemetry_and_status(self):
        r=m['classify'](sample(['AVAILABLE','NEW_STATE'],'0, GPU-0, H100, [N/A], 0\n1, GPU-1, H100, 0, 0'),256)
        self.assertEqual(r['free'],[]); self.assertEqual(r['unknown'],[0,1])
    def test_failed_or_mismatched_queries(self):
        for data in [sample(['AVAILABLE']).replace('METRICS_RC__0','METRICS_RC__1'),sample(['AVAILABLE'],'1, GPU-1, H100, 0, 0')]:
            with self.assertRaises(ValueError): m['classify'](data,256)
    def test_region_before_free_count(self):
        eu=dict(region='eu-frankfurt',host='eu',free=[])
        us=dict(region='us-east-ohio',host='us',free=list(range(8)))
        self.assertLess(m['sort_key'](eu,['eu','us-east']),m['sort_key'](us,['eu','us-east']))
    def test_output_excludes_extra_inventory_fields(self):
        from types import SimpleNamespace
        host = dict(host='example-accelerator', region='unknown', gpu='GAUDI',
                    address='192.0.2.1', vpn='example-private-network', owner='example-user')
        result = m['probe'](host, SimpleNamespace())
        self.assertEqual(set(result), {'host', 'region', 'gpu', 'error'})

if __name__ == '__main__':
    unittest.main()
