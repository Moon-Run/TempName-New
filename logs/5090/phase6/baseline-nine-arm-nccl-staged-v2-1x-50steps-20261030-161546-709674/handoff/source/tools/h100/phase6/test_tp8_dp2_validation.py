"""Portable CPU regressions: smoke-only evidence and compatible native caches."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import tp8_dp2_protocol as protocol
import tp8_dp2_train as train


def projection_row():
    names=[f'decoder.layers.{i}.{role}' for i in range(32)
           for role in ('self_attention.linear_proj','mlp.linear_fc2')]
    return dict(target_modules=names,forward_checks={n:dict(passed=True) for n in names})


def cache_fixture(root):
    contents={'native/kernel.cu':'kernel', 'deps/dependency.h':'dependency',
              'app/tp8_dp2_build.py':'recipe', 'app/tp8_dp2_train.py':'old controller'}
    for name,value in contents.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
    original=dict(protocol=protocol.PROTOCOL,model=protocol.MODEL,arms=list(protocol.ARMS),
                  files={n:protocol.sha(root/n) for n in contents},symlinks={},dependencies={'pinned':'revision'})
    original['package_id']=protocol.digest(original)
    protocol.write(root/'app/build-package.json',original)
    (root/'app/tp8_dp2_train.py').write_text('fixed controller')
    updated=copy.deepcopy(original)
    updated['build_reference']='app/build-package.json'
    for name in ('app/tp8_dp2_train.py','app/build-package.json'):
        updated['files'][name]=protocol.sha(root/name)
    seal(root,updated)
    return original,updated


def seal(root,info):
    info['package_id']=protocol.digest({k:v for k,v in info.items() if k!='package_id'})
    protocol.write(root/'package.json',info)


class ProjectionChecks(unittest.TestCase):
    def test_smoke_requires_complete_successful_named_coverage(self):
        protocol.validate_projection_checks(projection_row(),'smoke')
        for change in ('empty','missing','failed','wrong_module'):
            row=projection_row();name=row['target_modules'][0]
            if change=='empty': row['forward_checks']={}
            elif change=='missing': del row['forward_checks'][name]
            elif change=='failed': row['forward_checks'][name]['passed']=False
            else: row['forward_checks']['unrelated_module']=row['forward_checks'].pop(name)
            with self.subTest(change=change),self.assertRaises(ValueError):
                protocol.validate_projection_checks(row,'smoke')

    def test_timing_and_profile_require_empty_projection_audits(self):
        for mode in ('timing','profile'):
            row=projection_row();row['forward_checks']={}
            protocol.validate_projection_checks(row,mode)
            with self.assertRaisesRegex(ValueError,'leaked'):
                protocol.validate_projection_checks(projection_row(),mode)

    def test_unknown_mode_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unknown'):
            protocol.validate_projection_checks(projection_row(),'unknown')


class BuildCompatibility(unittest.TestCase):
    def test_controller_update_keeps_cache_but_changes_worker_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old,new=cache_fixture(root)
            self.assertNotEqual(old['package_id'],new['package_id'])
            protocol.verify_package(root)
            self.assertEqual(protocol.build_package_id(root,new),old['package_id'])
            stdout=io.StringIO()
            with contextlib.redirect_stdout(stdout):
                train.main(['--node-rank','0','--python','/opt/conda/bin/python','--package',str(root)])
            preview=json.loads(stdout.getvalue())
            self.assertEqual(preview['build_package_id'],old['package_id'])
            self.assertEqual(preview['package_id'],new['package_id'])
            self.assertFalse((root/'build').exists())

    def test_changed_native_dependency_or_recipe_cannot_reuse_cache(self):
        for name in ('native/kernel.cu','deps/dependency.h','app/tp8_dp2_build.py'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);old,new=cache_fixture(root)
                (root/name).write_text('changed')
                new['files'][name]=protocol.sha(root/name);seal(root,new)
                with self.assertRaisesRegex(ValueError,'Native build inputs differ'):
                    protocol.verify_package(root)

    def test_build_dependency_links_cannot_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old,new=cache_fixture(root)
            new['symlinks']['native/dependency']='../different-dependency'
            with self.assertRaisesRegex(ValueError,'Native build inputs differ'):
                protocol.build_package_id(root,new)

    def test_original_build_identity_must_be_authentic_to_its_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old,new=cache_fixture(root)
            old['package_id']='f'*64;protocol.write(root/'app/build-package.json',old)
            new['files']['app/build-package.json']=protocol.sha(root/'app/build-package.json');seal(root,new)
            with self.assertRaisesRegex(ValueError,'Invalid original build package identity'):
                protocol.verify_package(root)

    def test_legacy_package_still_uses_its_own_build_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old,_=cache_fixture(root)
            self.assertEqual(protocol.build_package_id(root,old),old['package_id'])


if __name__=='__main__': unittest.main()
