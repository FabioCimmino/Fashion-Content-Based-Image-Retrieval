from Updater import Updater
import os, sys, platform, subprocess
import warnings
warnings.filterwarnings("ignore")

def fileparts(fn):
	(dirName, fileName) = os.path.split(fn)
	(fileBaseName, fileExtension) = os.path.splitext(fileName)
	return dirName, fileBaseName, fileExtension


def imageHandler(bot, message, chat_id, local_filename):
	print(local_filename)
	# send message to user
	bot.sendMessage(chat_id, "Hi, please wait until the image is ready")
	# set command to start python script "main.py"
	cmd = "python main.py --query "+ local_filename
	# launch command
	subprocess.call(cmd,shell=True)
	# send back the manipulated image
	dirName, fileBaseName, fileExtension = fileparts(local_filename)
	new_fn = os.path.join(dirName, 'SIFT_result.jpg')
	colorResult = os.path.join(dirName, 'color_result.jpg')
	deepResult= os.path.join(dirName, 'DeepMethod_result.jpg')
	daisyResult= os.path.join(dirName, 'Daisy_result.jpg')
	bot.sendImage(chat_id, new_fn, "")
	bot.sendImage(chat_id, colorResult, "")
	bot.sendImage(chat_id, deepResult, "")
	bot.sendImage(chat_id, daisyResult, "")


if __name__ == "__main__":
	bot_id = '1211627469:AAGsJvk6ZiAsePOKuBNFsDrmG3eZaU4veic'
	updater = Updater(bot_id)
	updater.setPhotoHandler(imageHandler)
	updater.start()