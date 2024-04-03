from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QTreeWidgetItem, QTableWidgetItem, QMessageBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QFont, QPixmap, QImage, QColor
import pandas as pd
import numpy as np
from PIL import Image
from impro import process
import os
import subprocess
import glob
import shutil
import tempfile
from cryptography.fernet import Fernet
# import time

import tensorflow as tf
from tensorflow.keras.layers import Resizing

for filename in glob.glob('Resources\\tmp*'):
    shutil.rmtree(filename)
tempDir = tempfile.TemporaryDirectory(dir='Resources')


class finalBookToolbox(QMainWindow):
    def __init__(self):
        super(finalBookToolbox, self).__init__()



        loadUi("GUI.ui", self)
        self.initRenamer()
        self.initEncrypter()
        self.show()

    def initRenamer(self):
        self.renamerTW_df = pd.DataFrame()

        self.renamerInputB.clicked.connect(lambda: self.browseFolder(self.renamerInputLE, msg="Select Input Folder"))
        self.renamerInputB.clicked.connect(
            lambda: self.autoBrowseOutput(self.renamerInputLE, self.renamerOutputLE, add="_(renamed)"))
        self.renamerOutputB.clicked.connect(lambda: self.browseFolder(self.renamerOutputLE, msg="Select Output Folder"))

        self.fileLoaderB.clicked.connect(
            lambda: self.onRenamerLoadFiles(self.renamerTW, prefix=self.renamerPrefixCB.currentText()[1:],
                                            path=self.renamerInputLE.text()))
        self.renamerAutoB.clicked.connect(self.onRenamerAuto)
        self.renamerExecuteB.clicked.connect(self.onRenamerExecute)

        self.renamerOpenB.clicked.connect(self.onOpenRenamerExcel)
        self.renamerReloadB.clicked.connect(self.onReloadRenamerExcel)

        self.renamerTW.cellChanged.connect(self.onCellChanged)

        self.dialThresh.valueChanged.connect(self.dialLCDUpdater)
        self.reconfB.clicked.connect(lambda: self.sunjectFinder(self.renamerInputLE.text(), self.renamerPrefixCB.currentText()[1:]))
        self.resetB.clicked.connect(self.onResetButton)

    def initEncrypter(self):
        self.encrypterInputPathLog = []
        self.encrypterInputB.clicked.connect(lambda: self.browseFolder(self.encrypterInputLE, msg="Select Input Folder"))
        self.encrypterInputB.clicked.connect(
            lambda: self.autoBrowseOutput(self.encrypterInputLE, self.encrypterOutputLE, add="_(encrypted)"))
        self.encrypterInputB.clicked.connect(self.onEncrypterBrowseInput)
        self.encrypterOutputB.clicked.connect(lambda: self.browseFolder(self.encrypterOutputLE, msg="Select Output Folder"))
        self.encrypterExecuteB.clicked.connect(self.onEncrypterExecute)

    @staticmethod
    def popBox(msg):
        pop = QMessageBox()
        pop.setWindowTitle(" ")
        pop.setText(msg)
        pop.setIcon(QMessageBox.Warning)
        pop.exec_()

    def progress(self, value, text):
        self.progressBar.setValue(value)
        self.progressLabel.setText(f"{value}% - {text}")

    @staticmethod
    def openFile(path):
        subprocess.Popen(path, shell=True)

    def browseFolder(self, textInputWidget, msg="Select Folder"):
        folder_path = QFileDialog.getExistingDirectory(self, msg)
        textInputWidget.setText(folder_path)

    @staticmethod
    def autoBrowseOutput(textInputWidget_IN, textInputWidget_OUT, add):
        textInputWidget_OUT.setText(textInputWidget_IN.text() + add)

    def onCellChanged(self, row, column):
        text = self.renamerTW.item(row, column).text()
        self.renamerTW_df.iloc[row, column] = text

    def onRenamerLoadFiles(self, tableWidget, prefix, path):
        try:
            file_names = [i.split(".")[0] for i in os.listdir(path) if i.endswith(prefix)]
            self.renamerTW_df = pd.DataFrame()
            self.renamerTW_df["From"] = np.zeros(len(file_names))
            self.renamerTW_df["To"] = np.zeros(len(file_names))

            tableWidget.clear()
            tableWidget.setColumnCount(2)
            tableWidget.setHorizontalHeaderLabels(["From", "To"])

            self.progress(0, "")
            for i, name in enumerate(file_names):
                self.progress(int(i*100/len(file_names)), f"Loading File {name}{prefix}")
                tableWidget.setRowCount(i + 1)
                itemFrom = QTableWidgetItem(str(name))
                itemFrom.setFlags(Qt.ItemIsEnabled)
                itemFrom.setTextAlignment(Qt.AlignCenter)
                tableWidget.setItem(i, 0, itemFrom)
                itemTo = QTableWidgetItem(str(np.nan))
                itemTo.setTextAlignment(Qt.AlignCenter)
                tableWidget.setItem(i, 1, itemTo)

            self.sunjectFinder(path, prefix)


            self.progress(100, "Done Loading Files!")
        except:
            self.popBox("Error while loading files")

    def onRenamerAuto(self):
        self.worker = improWorkerThread(self.renamerTW_df, self.renamerInputLE.text(), self.renamerPrefixCB.currentText()[1:], self.dialThresh.value(),
                                        (int(self.leftCrop.text()), int(self.rightCrop.text())), (int(self.upCrop.text()), int(self.downCrop.text())))
        self.worker.start()
        self.worker.pred.connect(self.dfWriter)
        self.worker.done.connect(lambda: self.df2tw(self.renamerTW_df, self.renamerTW))
        self.worker.update_progress.connect(self.progress)

    def onRenamerExecute(self):
        if self.renamerTW_df["To"].nunique() < self.renamerTW_df.shape[0]:
            indx = self.renamerTW_df["To"].duplicated(keep=False)
            self.popBox(f"Duplicated name ERR\n{self.renamerTW_df[indx].index}")  #############FIX THIS
            return

        self.progress(0, "Checking directory!")
        if not os.path.isdir(self.renamerOutputLE.text()):
            os.makedirs(self.renamerOutputLE.text())
        for row, (F, T) in enumerate(zip(self.renamerTW_df["From"], self.renamerTW_df["To"])):
            self.progress(int(row*100/len(self.renamerTW_df["From"])), f"Renaming file {F}{self.renamerPrefixCB.currentText()[1:]} --> {T}{self.renamerPrefixCB.currentText()[1:]}")
            shutil.copy(f"{self.renamerInputLE.text()}/{F}{self.renamerPrefixCB.currentText()[1:]}",
                        f"{self.renamerOutputLE.text()}/{T}{self.renamerPrefixCB.currentText()[1:]}")
        self.progress(100, "Done Renaming Files!")

    def onOpenRenamerExcel(self):
        try:
            self.renamerTW_df.to_excel(f"{tempDir.name}\\renamerTW_df.xlsx", index=False)
        except:
            self.popBox("Error while saving *.xlsx file")
            return

        try:
            self.openFile(f"{tempDir.name}\\renamerTW_df.xlsx")
        except:
            self.popBox("Error while opening *.xlsx file")

    def dfWriter(self, predict, i):
        self.renamerTW_df.iloc[i, 1] = predict

    def df2tw(self, df, tableWidget):
        tableWidget.clear()
        tableWidget.setColumnCount(2)
        tableWidget.setHorizontalHeaderLabels(df.columns)

        self.progress(0, "")
        for row, (F, T) in enumerate(zip(df["From"], df["To"])):
            self.progress(int(row * 100 / len(df["From"])), f"Reloading Table {F},{T}")
            tableWidget.setRowCount(row + 1)
            itemFrom = QTableWidgetItem(str(F))
            itemFrom.setFlags(Qt.ItemIsEnabled)
            itemFrom.setTextAlignment(Qt.AlignCenter)
            itemTo = QTableWidgetItem(str(T))
            itemTo.setTextAlignment(Qt.AlignCenter)
            if len(str(T)) != int(self.seqLen.text()) or str(T).startswith("ERR"):
                itemTo.setBackground(QColor(255, 250, 150))
            tableWidget.setItem(row, 0, itemFrom)
            tableWidget.setItem(row, 1, itemTo)
        self.progress(100, "Done Reloading Table!")

    def onReloadRenamerExcel(self):
        try:
            temp_df = pd.read_excel(f"{tempDir.name}\\renamerTW_df.xlsx", converters={'From': str, 'To': str})
        except:
            self.popBox("No file to read")
            return
        try:
            self.df2tw(temp_df, self.renamerTW)
        except:
            self.popBox("Error while loading dataframe to table")

    # ENCRYPTER
    def loadFolder(self, startpath, tree):
        # print(startpath)
        for element in os.listdir(startpath):
            path_info = startpath + "\\" + element

            parent_itm = QTreeWidgetItem(tree, [os.path.basename(element).split(".")[0]])
            if os.path.isdir(path_info):
                self.loadFolder(path_info, parent_itm)
            #     parent_itm.setIcon(0, QIcon('Resources\\Images\\folder.png'))
            #     parent_itm.setFont(0, QFont("B Nazanine", 10, QFont.Bold))
            #     parent_itm.setExpanded(True)
            else:
                self.encrypterInputPathLog.append(path_info)
            #     parent_itm.setIcon(0, QIcon('Resources\\Images\\file.png'))

    def onEncrypterBrowseInput(self):
        self.encrypterTV.clear()
        self.encrypterTV.setHeaderLabel("")
        self.encrypterInputPathLog = []
        self.loadFolder(self.encrypterInputLE.text(), self.encrypterTV)
        # print(self.encrypterInputPathLog)

    def onEncrypterExecute(self):
        if len(self.encrypterInputPathLog) != 0:
            self.worker = encryptWorkerThread(self.encrypterInputPathLog, self.encrypterInputLE.text(), self.encrypterOutputLE.text())
            self.worker.start()
            self.worker.update_progress.connect(self.progress)
        else:
            self.popBox("No File to Encrypt")

    def dialLCDUpdater(self):
        thresh = self.dialThresh.value()
        self.lcdThresh.display(thresh)

    def sunjectFinder(self, path, prefix):
        file_names = [i.split(".")[0] for i in os.listdir(path) if i.endswith(prefix)]

        lC = int(self.leftCrop.text())
        rC = int(self.rightCrop.text())
        uC = int(self.upCrop.text())
        dC = int(self.downCrop.text())

        sub = process.pdf2ImgArray(path=f"{path}/{np.random.choice(file_names, 1)[0]}{prefix}", matrixSize=(4, 4))
        sub = process.edgeCutter(sub, 5, 30)
        sub = process.threshold(sub, self.dialThresh.value())
        sub = process.autoRotator(sub, rot_portion=5)
        sub = process.rescaleNegative(sub, 1 / 255)
        sub = process.borderCrop(sub, PMrange=200, hthresh=120, vthresh=200)
        sub = process.resizer(sub, size=(2520, 2000))

        sub = sub[uC:dC, lC:rC]

        sub = process.resizer(sub, size=(int(sub.shape[0]*350/sub.shape[1]), 350))

        png = Image.fromarray((1 - sub) * 250).convert("L")
        png.save(f"{tempDir.name}/img.png")

        self.pixmap = QPixmap(f"{tempDir.name}\\img.png")
        self.imgLabel.setPixmap(self.pixmap)

    def onResetButton(self):
        self.dialThresh.setValue(150)
        self.lcdThresh.display(150)
        self.leftCrop.setText("0")
        self.rightCrop.setText("2000")
        self.upCrop.setText("0")
        self.downCrop.setText("2520")

class improWorkerThread(QThread):
    def __init__(self, renamerTW_df, renamerInputLEtxt, renamerPrefixCBtxt, dialThreshValue, lrCord, udCord):
        super(improWorkerThread, self).__init__()

        self.renamerTW_df = renamerTW_df
        self.renamerInputLEtxt = renamerInputLEtxt
        self.renamerPrefixCBtxt = renamerPrefixCBtxt
        self.dialThreshValue = dialThreshValue
        self.lrCord = lrCord
        self.udCord = udCord

        self.model = tf.keras.models.load_model("tf_model/models/conv_model/model")
        self.resizerLayer = Resizing(20, 20)

    update_progress = pyqtSignal(int, str)
    pred = pyqtSignal(str, int)
    done = pyqtSignal()

    def run(self):
        self.update_progress.emit(0, "")
        for i, file_name in enumerate(self.renamerTW_df["From"]):
            self.update_progress.emit(int(i*100/len(self.renamerTW_df["From"])), f"Reading File {file_name}")
            #Step:1
            img = process.pdf2ImgArray(path=f"{self.renamerInputLEtxt}/{file_name}{self.renamerPrefixCBtxt}", matrixSize=(4, 4))
            if type(img) == str:
                self.update_progress.emit(int(i*100/len(self.renamerTW_df["From"])), f"ERR step1 for {file_name}")
                self.pred.emit(img, i)
                continue
            #Step:2
            img = process.edgeCutter(img, 5, 30)
            if type(img) == str:
                self.update_progress.emit(int(i*100/len(self.renamerTW_df["From"])), f"ERR step2 for {file_name}")
                self.pred.emit(img, i)
                continue
            #Step:3
            img = process.threshold(img, self.dialThreshValue)
            if type(img) == str:
                self.update_progress.emit(int(i*100/len(self.renamerTW_df["From"])), f"ERR step3 for {file_name}")
                self.pred.emit(img, i)
                continue
            #Step:4
            img = process.autoRotator(img, rot_portion=5)
            if type(img) == str:
                self.update_progress.emit(int(i*100/len(self.renamerTW_df["From"])), f"ERR step4 for {file_name}")
                self.pred.emit(img, i)
                continue
            #Step:5
            img = process.rescaleNegative(img, 1/255)
            if type(img) == str:
                self.update_progress.emit(int(i*100/len(self.renamerTW_df["From"])), f"ERR step5 for {file_name}")
                self.pred.emit(img, i)
                continue
            #Step:6
            img = process.borderCrop(img, PMrange=200, hthresh=120, vthresh=200)
            if type(img) == str:
                self.update_progress.emit(int(i*100/len(self.renamerTW_df["From"])), f"ERR step6 for {file_name}")
                self.pred.emit(img, i)
                continue
            #Step:7
            img = process.resizer(img, size=(2520, 2000))
            if type(img) == str:
                self.update_progress.emit(int(i*100/len(self.renamerTW_df["From"])), f"ERR step7 for {file_name}")
                self.pred.emit(img, i)
                continue
            #Step:8
            ##udCord=(262, 330), lrCord=(1750, 1905)
            img = process.findSubject(img, udCord=self.udCord, lrCord=self.lrCord)
            # os.makedirs(f"test_img/{file_name.split('.')[0]}", exist_ok=True)
            # png = Image.fromarray((1 - img) * 255).convert("L")
            # png.save(f"test_img/{file_name.split('.')[0]}/z{file_name.split('.')[0]}.png")
            if type(img) == str:
                self.update_progress.emit(int(i*100/len(self.renamerTW_df["From"])), f"ERR step8 for {file_name}")
                self.pred.emit(img, i)
                continue
            #Step:9
            imgs = process.segmentor(img)
            if type(imgs) == str:
                self.update_progress.emit(int(i*100/len(self.renamerTW_df["From"])), f"ERR step9 for {file_name}")
                self.pred.emit(imgs, i)
                continue

            #Step:10
            for j, img in enumerate(imgs):
                imgs[j] = np.expand_dims(imgs[j], axis=(0, -1))
                imgs[j] = self.resizerLayer(imgs[j])

            predict = ""
            for img in imgs:
                p = tf.argmax(tf.nn.softmax(self.model.predict(img, verbose=0)), axis=1)
                predict = predict + str(int(p))

            self.pred.emit(predict, i)

        self.update_progress.emit(100, "Done Reading Files!")
        self.done.emit()

class encryptWorkerThread(QThread):
    def __init__(self, path_log, input_path, output_path):
        super(encryptWorkerThread, self).__init__()
        self.fernetKey = Fernet(b'NDeVZN2769bnfxgI51FzSuXEGjNpIl1SvxauVNtUGKs=')
        self.path_log = path_log
        self.input_path = input_path
        self.output_path = output_path

    update_progress = pyqtSignal(int, str)

    def run(self):
        self.update_progress.emit(0, "Encryption Started")
        for i, path in enumerate(self.path_log):
            out = path.replace(self.input_path, self.output_path)
            file_name_index = out.rfind("\\")
            file_name = out[file_name_index:]
            path_without_name = out.replace(file_name, "")

            self.update_progress.emit(int(i*100/len(self.path_log)), f"Encrypting {file_name[1:]}")

            with open(path, 'rb') as file:
                orig = file.read()

            enc = self.fernetKey.encrypt(orig)

            if not os.path.isdir(path_without_name):
                os.makedirs(path_without_name)
            with open(out.split(".")[0] + ".finalbookpak", 'wb') as enc_file:
                enc_file.write(enc)
        self.update_progress.emit(100, "Encryption Done!")


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    MainWindow = finalBookToolbox()
    ret = app.exec_()
    tempDir.cleanup()
    sys.exit(ret)
