from setuptools import setup, find_packages
import subprocess
import sys
from pathlib import Path

def install_chatflare():
    chatflare_path = Path(__file__).parent / "ChatFlare"  # 如果实际目录是 "chatflare"，请修改这里为小写
    if (chatflare_path / "setup.py").exists():
        print("📦 Installing chatflare submodule...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-e", str(chatflare_path)
        ])
    else:
        print("⚠️  chatflare setup.py not found — did you forget to --recursive clone?")

from setuptools.command.install import install as InstallCommand
from setuptools.command.develop import develop as DevelopCommand

class CustomInstallCommand(InstallCommand):
    def run(self):
        InstallCommand.run(self)
        install_chatflare()

class CustomDevelopCommand(DevelopCommand):
    def run(self):
        DevelopCommand.run(self)
        install_chatflare()

def read_requirements(filename):
    return [
        line.strip()
        for line in Path(filename).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="insightagent",
    version="0.1.0",
    packages=find_packages(include=["ChatFlare*", "agent_card_server*"]),
    install_requires=read_requirements("requirements11.txt"),
    cmdclass={
        'install': CustomInstallCommand,
        'develop': CustomDevelopCommand,
    },
    python_requires=">=3.11",
)
