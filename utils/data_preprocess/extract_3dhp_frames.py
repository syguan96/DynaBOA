import subprocess
import os
import os.path as osp

def extract_video(vid_filename, output_folder):
    cmd = [
        'ffmpeg',
        '-i', vid_filename,
        f'{output_folder}/%06d.jpg',
        '-threads', '16'
    ]

    print(' '.join(cmd))
    try:
        subprocess.call(cmd)
    except OSError:
        print('OSError')

if __name__ == '__main__':

    for i in range(1, 9):
        for j in range(1, 3):
            for k in [0, 1, 2, 4, 5, 6, 7, 8]:
                vid = f'/data//mpi-inf-3dhp/S{i}/Seq{j}/imageSequence/video_{k}.avi'

                output_folder = f'/data//mpi-inf-3dhp/S{i}/Seq{j}/video_{k}'
                os.makedirs(output_folder, exist_ok=True)

                extract_video(vid, output_folder)