# -*- mode: python -*-

block_cipher = None


a = Analysis(['video_processing_v2.py'],
             pathex=['C:\\Users\\Sergey\\Documents\\GitHub\\cv_python\\tennis\\new'],
             binaries=None,
             datas=None,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['matplotlib', 'PyQt'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='video_processing_v2',
          debug=False,
          strip=False,
          upx=True,
          console=True )
