import classification.classification as classi
import features_extr.deep_method
import features_extr.SIFT
import features_extr.color.color as color
import optparse
import os, sys, platform, subprocess
import features_extr.Daisy 

def fileparts(fn):
	(dirName, fileName) = os.path.split(fn)
	(fileBaseName, fileExtension) = os.path.splitext(fileName)
	return dirName, fileBaseName, fileExtension

parser = optparse.OptionParser()

parser.add_option('-q', '--query',
    action="store", dest="query",
    help="query string", default="spam")

options, args = parser.parse_args()
img_path=options.query
dir_dataset='C:\\Users\\fabio\\Downloads\\dataset_category\\dataset_category'
#dir_dataset = 'C:\\Users\\pule\\Documents\\dataset_category'
classe_pred = classi.classify(img_path, dir_dataset)
dirName, fileBaseName, fileExtension = fileparts(img_path)
features_extr.SIFT.sift_extraction_bow(classe_pred,img_path, dir_dataset,dirName)
color.color(classe_pred,img_path,dir_dataset,dirName)
features_extr.deep_method.deep_method(classe_pred,img_path, dir_dataset,dirName)
features_extr.Daisy.daisy_extraction(classe_pred,img_path, dir_dataset,dirName)
