"""Synthetic filesystem checks only: no camera access, training, or model writes."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from prepare_field_data import prepare
from finetune_field import validate_field_dataset, record_epoch


class FieldDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.inventory = self.folder/'inventory.csv'
        self.output = self.folder/'prepared'
        self.rows = []
        for index, split in enumerate(('train','val','test')):
            image = self.folder/f'{split}.png'
            Image.new('RGB',(32,24),(40*index,10+index,100)).save(image)
            label = self.folder/f'{split}.txt'
            label.write_text(f'{index} 0.5 0.5 0.25 0.25\n',encoding='utf-8')
            self.rows.append({'image':image.name,'label':label.name,'wafer_id':f'synthetic-wafer-{index}',
                              'lot_id':'synthetic-lot','split':split,'reviewed':'yes','normal_verified':''})
        self.write_inventory()

    def tearDown(self):
        self.temporary.cleanup()

    def write_inventory(self):
        with self.inventory.open('w',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)

    def prepared(self):
        prepare(self.inventory,self.output)
        return json.loads((self.output/'audit.json').read_text(encoding='utf-8'))

    def write_audit(self,audit):
        (self.output/'audit.json').write_text(json.dumps(audit),encoding='utf-8')

    def test_valid_synthetic_snapshot_and_identity_boundary(self):
        self.prepared()
        audit=validate_field_dataset(self.output)
        self.assertEqual(len(audit['records']),3)
        self.assertEqual(audit['physical_identity_basis'],'User-supplied wafer_id, not independently measured')

    def test_unreviewed_inventory_rejected_before_output(self):
        self.rows[1]['reviewed']='no'; self.write_inventory()
        with self.assertRaisesRegex(ValueError,'reviewed must be yes'):
            prepare(self.inventory,self.output)
        self.assertFalse(self.output.exists())

    def test_same_physical_wafer_cross_split_rejected(self):
        self.rows[1]['wafer_id']=self.rows[0]['wafer_id']; self.write_inventory()
        with self.assertRaisesRegex(ValueError,'physical wafer appears in two splits'):
            prepare(self.inventory,self.output)
        self.assertFalse(self.output.exists())

    def test_decoded_duplicate_across_filenames_rejected(self):
        Image.open(self.folder/'train.png').save(self.folder/'val.png')
        with self.assertRaisesRegex(ValueError,'Duplicate decoded image'):
            prepare(self.inventory,self.output)

    def test_empty_label_needs_normal_review(self):
        (self.folder/'val.txt').write_text('',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'empty label requires'):
            prepare(self.inventory,self.output)
        self.rows[1]['normal_verified']='yes'; self.write_inventory()
        self.prepared()
        self.assertEqual(validate_field_dataset(self.output)['counts']['val']['normal_images'],1)

    def test_nonfinite_label_rejected(self):
        (self.folder/'train.txt').write_text('0 nan 0.5 0.25 0.25',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Invalid label row'):
            prepare(self.inventory,self.output)

    def test_unreviewed_extra_image_after_audit_rejected(self):
        self.prepared()
        Image.new('RGB',(32,24),(250,240,230)).save(self.output/'images/train/unreviewed.png')
        with self.assertRaisesRegex(RuntimeError,'Unreviewed or missing'):
            validate_field_dataset(self.output)

    def test_label_changed_after_audit_rejected(self):
        audit=self.prepared()
        (self.output/audit['records'][0]['label']).write_text('0 0.5 0.5 0.5 0.5',encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,'changed after audit'):
            validate_field_dataset(self.output)

    def test_audit_cannot_route_files_outside_dataset(self):
        audit=self.prepared(); audit['files'][0]['path']='../train.png'; self.write_audit(audit)
        with self.assertRaisesRegex(RuntimeError,'changed after audit'):
            validate_field_dataset(self.output)

    def test_unaudited_record_or_split_metadata_rejected(self):
        audit=self.prepared(); audit['records'][1]['wafer_id']=audit['records'][0]['wafer_id']; self.write_audit(audit)
        with self.assertRaisesRegex(RuntimeError,'wafer crosses'):
            validate_field_dataset(self.output)

    def test_field_yaml_changed_after_audit_rejected(self):
        self.prepared()
        with (self.output/'data.yaml').open('a',encoding='utf-8') as stream:
            stream.write('\n# changed after snapshot\n')
        with self.assertRaisesRegex(RuntimeError,'audit required'):
            validate_field_dataset(self.output)

    def test_final_eval_is_not_a_training_epoch(self):
        history=[]
        trainer=SimpleNamespace(epoch=0,field_steps=5,metrics={'metrics/mAP50-95(B)':.25},in_final_eval=False)
        self.assertEqual(record_epoch(trainer,history)['epoch'],1)
        trainer.in_final_eval=True
        self.assertIsNone(record_epoch(trainer,history))
        self.assertEqual(len(history),1)
        trainer.in_final_eval=False
        with self.assertRaisesRegex(RuntimeError,'Duplicated'):
            record_epoch(trainer,history)


if __name__=='__main__':
    unittest.main(verbosity=2)
