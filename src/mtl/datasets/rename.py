import os
from pathlib import Path

def main():
    """Rename images to numbers: TODO check consistency of rgb and labels"""
    print("in")
    path = Path(__file__).parents[3]
    path = os.path.join(path, 'data/exp/test')
    path_rgb = os.path.join(path, 'rgb')
    path_lab = os.path.join(path, 'semseg')
    files_r=[name for name in os.listdir(path_rgb) if os.path.isfile(os.path.join(path_rgb, name))]
    files=[name for name in os.listdir(path_rgb) if os.path.isfile(os.path.join(path_rgb, name))]

    for i,file in enumerate(files):
        f=file.split('_')[1]
        fi=f.split('.')[0]
        num=int(fi)
        os.rename(path_rgb+ '/rgb_'+str(num)+'.png', path_rgb+'/'+ str(i) + ".png")
        os.rename(path_lab+ '/label_'+str(num)+'.png', path_lab +'/'+ str(i) + ".png")
    print("done")


if __name__ == '__main__':
    print("main")
    main()
