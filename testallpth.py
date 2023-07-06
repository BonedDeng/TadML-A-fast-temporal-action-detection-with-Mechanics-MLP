import os
import glob
import subprocess
import shlex

ckptpath = './ckpt/thumos_i3d_wavefpn8/'
assert os.path.isdir(ckptpath), 'ckpt is not exist'
ckpt_file_list = sorted(glob.glob(os.path.join(ckptpath, '*.pth.tar')))

def writelog(ckpt_file):
    command = 'python ./inference.py ./hyps/thumos_i3d.yaml {}'.format(ckpt_file)
    back = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()
    # print("back0----", back[0].decode())  # 
    # print("back1----", back[1].decode())  #
    with open('.'+str(str(ckpt_file).split('.')[1])+'.log','a',encoding='utf-8') as f:
        f.write(back[0].decode())
        f.write(back[1].decode())
        f.close()
for i, j in enumerate(ckpt_file_list):
    writelog(j)
    print("log{}already has write success!".format(i))