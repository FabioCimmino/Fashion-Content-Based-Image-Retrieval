# import the necessary packages
from ColorDescriptor import ColorDescriptor
import glob, cv2, os


# initialize the color descriptor
cd = ColorDescriptor((8, 8, 8))

classi = os.listdir('C:\\Users\\fabio\\Downloads\\dataset_category\\dataset_category')
classi.sort()

for classe in classi:
    # open the output index file for writing
    output = open('descriptor/index_' + classe + '_dataset.csv', "w")
    

    # use glob to grab the image paths and loop over them
    for imagePath in glob.glob('C:\\Users\\fabio\\Downloads\\dataset_category\\dataset_category\\' + classe + "/*.jpg"):
        # extract the image ID (i.e. the unique filename) from the image
        # path and load the image itself
        imageID = imagePath[imagePath.rfind("/") + 1:]
        image = cv2.imread(imagePath)

        # describe the image
        features = cd.describe(image)

        # write the features to file
        features = [str(f) for f in features]
        output.write("%s,%s\n" % (imageID, ",".join(features)))

    # close the index file
    output.close()