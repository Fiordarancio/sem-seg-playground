# Create list file for a given dataset. Images and labels must have the 
# same name, while images have .jpg extension and labels have the .png one.
#------------------------------------------------------------------------
import argparse
import os 

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True, type=str,
                    help='Name of the split to be built')
parser.add_argument('--img_dir', required=True, type=str,
                    help='Path to the images folder')
parser.add_argument('--lab_dir', required=False, type=str,
                    help='Path to the annotation folder')
args = parser.parse_args()

num_img = len(os.listdir(args.img_dir))

if num_img == 0:
  raise ValueError('%s is empty' %(args.img_dir))
if args.lab_dir is not None:
  num_lab = len(os.listdir(args.lab_dir))
  if num_img != num_lab:
    raise ValueError('%s does not contain the same objects of %s' % (args.img_dir, args.lab_dir))

if args.img_dir.strip().endswith('/'):
  args.img_dir = args.img_dir.strip()[:-1]
if args.lab_dir is not None:
  if args.lab_dir.strip().endswith('/'):
    args.lab_dir = args.lab_dir.strip()[:-1]

list_name = '%s/list/%s.lst' % (os.getcwd(), args.name)
images = os.listdir(args.img_dir)
images.sort() # sort alphabetically 

if args.lab_dir is not None:
  labels = os.listdir(args.lab_dir)
  labels.sort()
  with open(list_name, 'w') as lf:
    for (x,y) in zip(images, labels):
      lf.write('%s/%s %s/%s\n' % (args.img_dir, x, args.lab_dir, y))
else:
  with open(list_name, 'w') as lf:
    for x in images:
      lf.write('%s/%s\n' % (args.img_dir, x))

print('Got %d examples listed at %s' % (num_img, list_name))