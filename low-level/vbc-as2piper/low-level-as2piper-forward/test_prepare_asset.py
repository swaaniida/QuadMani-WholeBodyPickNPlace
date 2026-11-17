"""Shared-directory regression: never chmod another user's existing asset."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from prepare_asset import prepare


class PrepareAssetTest(unittest.TestCase):
    def test_create_skip_and_replace_without_path_chmod(self):
        original=Path(__file__).parent/'resources/robots/as2piper/urdf/as2piper_umi_v7_mesh_seated_plate_only.urdf'
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/original.name
            shutil.copyfile(str(original),str(source))
            with patch.object(Path,'chmod',side_effect=PermissionError('other owner')):
                output=prepare(source)
                expected=output.read_bytes()
                self.assertEqual(output.stat().st_mode & 0o777,0o664)
                before=output.stat()
                prepare(source)
                self.assertEqual(before.st_ino,output.stat().st_ino)
                self.assertEqual(before.st_mtime_ns,output.stat().st_mtime_ns)
                output.write_bytes(b'outdated asset')
                prepare(source)
                self.assertEqual(expected,output.read_bytes())
                self.assertNotEqual(before.st_ino,output.stat().st_ino)
                self.assertEqual(list(Path(directory).glob('*.tmp')),[])


if __name__=='__main__':unittest.main()
