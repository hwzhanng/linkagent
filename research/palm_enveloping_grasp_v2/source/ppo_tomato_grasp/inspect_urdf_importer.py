"""Print the URDF importer API exposed by the installed Isaac Sim runtime."""
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
import omni.kit.commands

_, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
print(type(cfg), sorted(name for name in dir(cfg) if name.startswith("set_")), flush=True)
app.close()
