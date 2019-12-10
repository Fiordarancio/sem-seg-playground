from PIL import Image
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dir', 
                    help='Directory of visualization (fullpath)',
                    required=True)
parser.add_argument('--saveOutput',
                    help='Specify if you want to save the overlaid image',
                    required=False,
                    action='store_true')
parser.add_argument('--showLimit',
                    help='Number of maximum predictions to show. Use -1 to show all (default: None)',
                    required=False,
                    type=int)    
parser.add_argument('--name',
                    help='Name or information about the generating network',
                    required=False)
args = parser.parse_args()

visDir = args.dir
isSave = args.saveOutput
sLimit = args.showLimit
netName= args.name

print('Visualizing images in ', visDir)
numFiles = len(os.listdir(visDir))
if sLimit == -1:
    sLimit = numFiles // 2 

print('Found ', numFiles, ' files (', (numFiles//2), ', couples )')

for i in range(numFiles//2):
    imgFile = '%06d_image.png' % (i)
    labFile = '%06d_prediction.png' % (i)
    print('Opening image ', imgFile, ' with prediction ', labFile)

    imgFile = os.path.join(visDir, imgFile)
    labFile = os.path.join(visDir, labFile)
    background = Image.open(imgFile)
    annotation = Image.open(labFile)

    background = background.convert('RGBA') 
    annotation = annotation.convert('RGBA')

    overlay = Image.blend(background, annotation, 0.65)
    if isSave:
        overlay.save(os.path.join(visDir, '%05d_overlay.png' % (i)))
    
    if sLimit is not None and i < sLimit:
        # plt.imshow(annotation, cmap=plt.cm.get_cmap('cubehelix', 34))
        plt.imshow(overlay)
        if netName is not None:
            plt.title(netName)
        # plt.colorbar(ticks=range(34), label='Labels')
        plt.show()

print('Images overlay complete')