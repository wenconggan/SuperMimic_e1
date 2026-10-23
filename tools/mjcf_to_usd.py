"""Offline MJCF -> USD conversion for the original kit-less Isaac Lab runtime."""

import argparse
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["e1"], default="e1")
    args = parser.parse_args()
    os.environ["SUPERMIMIC_ROBOT"] = args.robot
    from supermimic.robot import MJCF_PATH, USD_DIR

    src = Path(MJCF_PATH)
    if args.robot == "e1":
        # Isaac Lab supplies the floor and implicit PD drives; leave the user's
        # standalone asset intact and convert a robot-only copy.
        tree = ET.parse(src)
        root = tree.getroot()
        compiler = root.find("compiler")
        compiler.set("meshdir", str((src.parent / compiler.get("meshdir", ".")).resolve()))
        for tag in ("contact", "actuator", "sensor", "keyframe"):
            for element in root.findall(tag):
                root.remove(element)
        for world in root.findall("worldbody"):
            for element in list(world):
                if element.tag != "body":
                    world.remove(element)
        src = REPO / "data" / "e1_description" / "e1_21dof.xml"
        src.parent.mkdir(parents=True, exist_ok=True)
        tree.write(src, encoding="unicode")

    from pxr import Plug
    import isaacsim.asset.importer.utils.impl as importer_utils
    Plug.Registry().RegisterPlugins(str(Path(importer_utils.__file__).parent / "schemas"))
    from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

    cfg = MJCFImporterConfig(mjcf_path=str(src), usd_path=USD_DIR)
    asset_path = MJCFImporter(cfg).import_mjcf()
    if args.robot == "e1":
        # The importer chooses a suffixed directory on repeated conversions.
        # Record the actual output so the next training process uses that asset.
        relative = Path(asset_path).resolve().relative_to(Path(USD_DIR).resolve())
        (Path(USD_DIR) / "asset_path.txt").write_text(str(relative) + "\n")
    print(asset_path)


if __name__ == "__main__":
    main()
