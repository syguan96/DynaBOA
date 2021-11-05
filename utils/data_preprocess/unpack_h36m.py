import os
import glob

SAVEPATH_H36M = ''

os.makedirs(SAVEPATH_H36M)
files = glob.glob('downloader/*/subject/s*/*.tgz')
for f in files:
    cmd = f'tar -xvf {f} -C {SAVEPATH_H36M}'
    os.system(cmd)
