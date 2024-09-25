# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import copy_metadata, collect_data_files
import torch
import inspect

datas = []
datas += collect_data_files('torch')
datas += copy_metadata("packaging")
datas += copy_metadata("tokenizers")
datas += copy_metadata('transformers')
datas += copy_metadata('importlib_metadata')
datas += copy_metadata('tqdm')
datas += copy_metadata('regex') 
datas += copy_metadata('requests')
datas += copy_metadata('packaging')
datas += copy_metadata('filelock')
datas += copy_metadata('numpy')
datas += copy_metadata('huggingface-hub')
datas += copy_metadata('safetensors')
datas += copy_metadata('pyyaml')
datas += collect_data_files('yapf_third_party')
block_cipher = None



a = Analysis(
    ['D:\GroundingDINO_live\Open-GroundingDino\main_LoRA.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['matplotlib.backends.backend_tkagg','requests','numpy','huggingface-hub','safetensors','pyyaml','torch'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='groundingDINO_lora_training',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='groundingDINO_lora_training',
)
