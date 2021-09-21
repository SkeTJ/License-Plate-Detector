import cv2
import pytesseract
import numpy as np
import os
import sys
import csv

#UI
from kivymd.app import MDApp
from kivy.lang.builder import Builder
from kivymd.toast import toast
from kivymd.uix.filemanager import MDFileManager
from kivy.uix.image import Image
from kivy.uix.screenmanager import Screen
from kivy.uix.button import Button
from kivymd.uix.button import MDIconButton
from kivy.uix.gridlayout import GridLayout
from kivymd.uix.textfield import MDTextField
from kivymd.uix.button import MDRectangleFlatButton
from kivymd.uix.label import MDLabel

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class ImageScreenList(Screen):
    def __init__(self, **kwargs):
        super(ImageScreenList, self).__init__(**kwargs)
        self.TitleLabel()
        self.IconButton()
        self.FileManager()
        self.ConsoleText()

        #Get original file path
        self.originalDirectory = os.getcwd()

    #Setup Title of the Application
    def TitleLabel(self):
        self.titleLbl = MDLabel()
        self.titleLbl.text = "License Plate Detection"
        self.titleLbl.halign = "center"
        self.titleLbl.font_size = 24
        self.titleLbl.pos_hint = {"center_x": .5, "center_y": 0.98}

        self.add_widget(self.titleLbl)

    #Setup Open File Button
    def IconButton(self):
        self.iconBtn = MDIconButton()
        self.iconBtn.icon = "file-upload"
        self.iconBtn.md_bg_color = "orange"
        self.iconBtn.pos_hint = {"center_x": .95, "center_y": .125}
        self.iconBtn.bind(on_press = self.OpenFileImage)
        self.add_widget(self.iconBtn)

    #Setup File Manager
    def FileManager(self):
        self.manager_open = False
        self.file_manager = MDFileManager(
            exit_manager = self.exit_manager,
            select_path = self.select_path,
            preview = False,
        )

        self.file_manager.ext = [".png", ".jpg"]
    
    #Open File Dialog
    def OpenFileImage(self, args):
        self.file_manager.show('/')
        self.manager_open = True

    #Get path of image and display it in the UI
    def select_path(self, path):
        self.exit_manager()
        self.originalPath = path
        toast(path)
        self.DisplayImage(path)

    #Exit File Dialog  
    def exit_manager(self, *args):
        self.manager_open = False
        self.file_manager.close()

    #Display Image on Screen
    def DisplayImage(self, path):
        self.displayImg = Image()
        self.displayImg.source = path
        self.displayImg.size_hint = 0.5, 0.5
        self.displayImg.keep_ratio = True
        self.displayImg.pos_hint = {"center_x": .5, "center_y": .65}
        
        self.add_widget(self.displayImg)
        self.consoleText.text = path
        self.iconBtn.icon = "restart"
        self.iconBtn.bind(on_press = self.RestartApplication)   
        self.StartPreProcessing()

    #For displaying text onto the console
    def ConsoleText(self):
        self.consoleText = MDTextField()
        self.consoleText.size_hint = 0.8, 0.08
        self.consoleText.pos_hint = {"center_x": .5, "center_y": .25}
        self.consoleText.halign = "center"
        self.consoleText.font_size = 22

        self.add_widget(self.consoleText)

    #Let user to decide whether to start the process
    def StartPreProcessing(self):
        self.startBtn = MDRectangleFlatButton()
        self.startBtn.text = "Start"
        self.startBtn.font_size = 22
        self.startBtn.opacity = 1
        self.startBtn.disabled = False
        self.startBtn.size_hint = 0.1, 0.05
        self.startBtn.pos_hint = {"center_x": .5, "center_y": .15}
        self.startBtn.bind(on_press = self.StartProcessing)

        self.add_widget(self.startBtn)

    #Start image processing and store them in a directory
    def StartProcessing(self, args):
        self.consoleText.text = ""
        
        #Create Directory
        currDirectory = os.getcwd()
        self.newDirectory = os.path.join(currDirectory, r'Processed Images')
        if not os.path.exists(self.newDirectory):
            os.makedirs(self.newDirectory)
        else:
            list(map(os.unlink, (os.path.join(self.newDirectory, files) for files in os.listdir(self.newDirectory))))

        #Go to new directory
        os.chdir(self.newDirectory)
        
        #Get image source
        img = cv2.imread(str(self.displayImg.source))

        #Grayscale image
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imwrite('Gray.jpg', grayImg)
        self.grayText = pytesseract.image_to_string(grayImg).replace('\f','')

        #Resize Image
        resizeImg = cv2.resize(grayImg, None, fx = 5, fy = 5, interpolation = cv2.INTER_CUBIC)

        cv2.imwrite('Resize.jpg', resizeImg)
        self.resizeText = pytesseract.image_to_string(resizeImg).replace('\f','')

        #Blur Image/Noise
        blurImg = cv2.medianBlur(resizeImg, 5)

        cv2.imwrite('Blur.jpg', blurImg)
        self.blurText = pytesseract.image_to_string(blurImg).replace('\f','')

        #Threshold Image
        thresholdImg = cv2.threshold(resizeImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        cv2.imwrite('Threshold.jpg', thresholdImg)
        self.thresholdText = pytesseract.image_to_string(thresholdImg).replace('\f','')

        #Threshold Blur
        thresholdBlurImg = cv2.threshold(blurImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        cv2.imwrite('Threshold Blur.jpg', thresholdBlurImg)
        self.thresholdBlurText = pytesseract.image_to_string(thresholdBlurImg).replace('\f','')

        #Invert Threshold Image
        w, h = thresholdImg.shape[:2]
        if cv2.countNonZero(thresholdImg) > ((w * h) // 2):
            self.invertedText = ""
        else:
            invertImg = cv2.bitwise_not(thresholdImg)
            cv2.imwrite('Inverted.jpg', invertImg)
            self.invertedText = pytesseract.image_to_string(invertImg).replace('\f','')

        #Force Invert Image
        forceInvertImg = cv2.bitwise_not(thresholdImg)
        cv2.imwrite('Force Inverted.jpg', forceInvertImg)
        self.forceInvertedText = pytesseract.image_to_string(forceInvertImg).replace('\f','')

        #Invert Threshold Blur Image
        w, h = thresholdBlurImg.shape[:2]
        if cv2.countNonZero(thresholdBlurImg) > ((w * h) // 2):
            self.invertedBlurText = ""
        else:
            invertBlurImg = cv2.bitwise_not(thresholdBlurImg)
            cv2.imwrite('Inverted Blur.jpg', invertBlurImg)
            self.invertedBlurText = pytesseract.image_to_string(invertBlurImg).replace('\f','')

        #Morphological Operations (Dilate, Eroded & Open Image)
        kernel = np.ones((5, 5), np.uint8)
        dilateImg = cv2.dilate(resizeImg, kernel, iterations = 1)
        erodeImg = cv2.erode(resizeImg, kernel, iterations = 1)
        openImg = cv2.morphologyEx(resizeImg, cv2.MORPH_OPEN, kernel)

        cv2.imwrite('Dilate.jpg', dilateImg)
        cv2.imwrite('Erode.jpg', erodeImg)
        cv2.imwrite('Open.jpg', openImg)

        self.dilateText = pytesseract.image_to_string(dilateImg).replace('\f','')
        self.erodeText = pytesseract.image_to_string(erodeImg).replace('\f','')
        self.openText = pytesseract.image_to_string(openImg).replace('\f','')

        #Canny Edge Detection
        cannyImg = cv2.Canny(resizeImg, 100, 200)

        cv2.imwrite('Canny.jpg', cannyImg)
        self.cannyText = pytesseract.image_to_string(cannyImg).replace('\f','')

        #Combined Image
        blurGrayImg = cv2.medianBlur(grayImg, 5)
        resizeGrayImg = cv2.resize(blurGrayImg, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
        thresholdGrayImg = cv2.threshold(resizeGrayImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        wG, hG = thresholdGrayImg.shape[:2]

        #Invert the image
        if cv2.countNonZero(thresholdGrayImg) > ((wG * hG) // 2):
            #If detected there are more black in the picture, don't alter the image
            cv2.imwrite('Combined.jpg', thresholdGrayImg)
            self.combinedText = pytesseract.image_to_string(thresholdGrayImg).replace('\f','')
        else:
            #If detected there are more white in the picture, invert the colours
            invertGrayImg = cv2.bitwise_not(thresholdGrayImg)
            cv2.imwrite('Combined.jpg', invertGrayImg)
            self.combinedText = pytesseract.image_to_string(invertGrayImg).replace('\f','')

        #Cropped Image Only   
        try:
            #Import Haar Cascade XML file for Russian car plate numbers
            carplate_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

            licensePlate = resizeImg.copy()
            licenseRectInImg = carplate_haar_cascade.detectMultiScale(licensePlate,scaleFactor = 1.1, minNeighbors = 5)
            
            for x,y,w,h in licenseRectInImg:
                #Overlay rectangle onto the license plate (Debugging purposes)
                cv2.rectangle(licensePlate, (x, y), (x + w, y + h), (0, 255, 0), 2)

            for x,y,w,h in licenseRectInImg:
                #Will cropped to the license plate
                croppedImg = img[y + 15: y + h - 10, x + 15: x + w - 20]

            cv2.imwrite('Cropped.jpg', croppedImg)
            self.croppedText = pytesseract.image_to_string(croppedImg).replace('\f','')
        except:
            self.croppedText = ""
            #print("Couldn't find any license plate and crop it")

        self.consoleText.text = "Done processing all the image!"
        self.startBtn.disabled = True
        self.startBtn.opacity = 0

        #Intiate Buttons and Caption
        self.ShowCaptions()
        self.ShowSave()
        self.ShowPreviousNext()
        
    #Display text which correspond to the image that is being display
    def ShowCaptions(self):
        self.captionLbl = MDLabel()
        self.captionLbl.text = ""
        self.captionLbl.halign = "center"
        self.captionLbl.font_size = 24
        self.captionLbl.pos_hint = {"center_x": .5, "center_y": .2}

        self.add_widget(self.captionLbl)

    #Initiate button for saving to CSV
    def ShowSave(self):
        self.saveBtn = MDRectangleFlatButton()
        self.saveBtn.text = "Save"
        self.saveBtn.font_size = 22
        self.saveBtn.opacity = 1
        self.saveBtn.disabled = False
        self.saveBtn.size_hint = 0.1, 0.05
        self.saveBtn.pos_hint = {"center_x": .5, "center_y": .04}
        self.saveBtn.bind(on_press = self.SaveToCSV)

        self.add_widget(self.saveBtn)

    #Save data to CSV
    def SaveToCSV(self, args):
        #Change to original directory
        os.chdir(self.originalDirectory)

        #Get Original Image Name
        base = os.path.basename(self.originalPath)

        #Header for CSV File
        fieldHeader = ['Image Name', 'Blur', 'Canny', 'Combined', 'Cropped', 'Dilate',
                       'Erode', 'Force Inverted', 'Gray', 'Inverted Blur',
                       'Inverted', 'Open', 'Resize', 'Threshold Blur', 'Threshold']

        #Get the field name to write to CSV File
        fields = [{'Image Name': base, 'Blur': self.blurText, 'Canny': self.cannyText, 'Combined': self.combinedText,
                   'Cropped': self.croppedText, 'Dilate': self.dilateText, 'Erode': self.erodeText, 'Force Inverted': self.forceInvertedText, 'Gray': self.grayText, 
                   'Inverted Blur': self.invertedBlurText, 'Inverted': self.invertedText, 'Open': self.openText,
                   'Resize': self.resizeText, 'Threshold Blur': self.thresholdBlurText, 'Threshold': self.thresholdText}]
        #Check whether file exists and if not create a new CSV File
        if not os.path.isfile('data.csv'):
            csvFile = 'data.csv'
            with open(csvFile, 'w', newline="") as newCSVFile:
                writer = csv.DictWriter(newCSVFile, fieldnames = fieldHeader)
                writer.writeheader()
                writer.writerows(fields)
                toast('Saved!')
        else:
            #Append instead if there is a CSV file
            with open('data.csv', 'a') as appendFile:
                writer = csv.DictWriter(appendFile, fieldnames = fieldHeader)
                writer.writerows(fields)
                toast('Saved!')

    #Show buttons to scroll through the different processed image            
    def ShowPreviousNext(self):
        #Initiate image counter here instead
        #as to not be disturb by for loop
        self.imgCount = 0
        
        nextBtn = MDIconButton()
        nextBtn.icon = "chevron-right"
        nextBtn.md_bg_color = "orange"
        nextBtn.pos_hint = {"center_x": .55, "center_y": .125}
        nextBtn.bind(on_press = self.NextImage)
        self.add_widget(nextBtn)

        previousBtn = MDIconButton()
        previousBtn.icon = "chevron-left"
        previousBtn.md_bg_color = "orange"
        previousBtn.pos_hint = {"center_x": .45, "center_y": .125}
        previousBtn.bind(on_press = self.PreviousImage)
        self.add_widget(previousBtn)

    #Go to next processed image
    def NextImage(self, args):
        img = []
        imgPath = self.newDirectory
        if len(img) <= 0:
            for i in os.listdir(imgPath):
                if ".jpg" in i:
                    img.append(imgPath + "/" + i)

        #Instead of crashing the application, 
        #instead allow user to keep using the program
        #when the list goes out of the index range
        try:
            #Increase increment and get file name for captioning by splitting the head
            self.imgCount = self.imgCount + 1
            base = os.path.basename(str(img[self.imgCount]))

            #Display the image, captions and text found from the image
            self.displayImg.source = img[self.imgCount]
            self.captionLbl.text = base
            self.consoleText.text = pytesseract.image_to_string(img[self.imgCount]).replace('\f','')       
        except:
            toast("End of list")

    #Go to previous processed image
    def PreviousImage(self, args):
        img = []
        imgPath = self.newDirectory
        if len(img) <= 0:
            for i in os.listdir(imgPath):
                if ".jpg" in i:
                    img.append(imgPath + "/" + i)

        #Instead of crashing the application, 
        #instead allow user to keep using the program
        #when the list goes out of the index range
        try:
            #Decrease increment and get file name for captioning by splitting the head
            self.imgCount = self.imgCount - 1
            base = os.path.basename(str(img[self.imgCount]))

            #Display the image, captions and text found from the image       
            self.displayImg.source = img[self.imgCount]
            self.captionLbl.text = base
            self.consoleText.text = pytesseract.image_to_string(img[self.imgCount]).replace('\f','')  
        except:
            toast("End of list")

    #Restart application
    def RestartApplication(self, args):
        os.chdir(self.originalDirectory)
        print(os.path.abspath(__file__))
        os.execl(sys.executable, os.path.abspath(__file__), *sys.argv) 

kv = '''
<ImageScreenList>:
    Image:
        id: originalImage
        
'''

class Main(MDApp):
    def build(self):
        self.title = "LPD Police"
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "DeepOrange"

        return ImageScreenList()

if __name__ == '__main__':
    Main().run()
