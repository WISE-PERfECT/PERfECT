'''
Brief: 
Author: Xinyu Tian
Github: https://github.com/XITechs
CreateTime: 2021-11-03 16:44:31
LastEditTime: 2021-11-05 15:32:19
Description: 
'''

import sys
from PyQt5 import QtCore, QtSerialPort
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QDialog, QDialogButtonBox, QFormLayout, QComboBox, QHBoxLayout, QLineEdit, QLabel, QPushButton, QTextEdit, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtSerialPort import QSerialPortInfo

import time
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
import os.path
import pandas as pd

import collections


class COMportConfig(QDialog):

    def __init__(self):
        super(COMportConfig,self).__init__()
        
        self.COMport_cb = QComboBox()
        self.baud_cb = QComboBox()
        
        for port in QSerialPortInfo.availablePorts():
            self.COMport_cb.addItem(port.portName())
        
        for baudr in QSerialPortInfo.standardBaudRates():
            self.baud_cb.addItem(str(baudr),baudr)
        
        bBox = QDialogButtonBox()
        bBox.setOrientation(Qt.Horizontal)
        bBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
        bBox.accepted.connect(self.accept)
        bBox.rejected.connect(self.reject)
        
        layD = QFormLayout(self)
        layD.addRow("Port:", self.COMport_cb)
        layD.addRow("Baud rate:", self.baud_cb)
        layD.addRow(bBox)
        self.setFixedSize(self.sizeHint())
    
    def get_results(self):
        return self.COMport_cb.currentText(), self.baud_cb.currentData(), self.baud_cb.currentText()

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("PERfECTtest")

        self.command_line = QLineEdit()
        self.send_button = QPushButton(
            text = "Send",
            clicked = self.send
        )

        self.output_serial = QTextEdit(readOnly = True)
        self.connect_button = QPushButton(
            text = "Connect",
            checkable = True,
            toggled = self.on_toggled
        )
        
        self.configure_button = QPushButton(
            text = "Configure port and baud",
            clicked = self.COMport
        )

        self.RC_vdsInput = QLineEdit("-600")
        self.RC_pulseHInput = QLineEdit("800")
        self.RC_send_btn = QPushButton(
            text = "Start RC",
            clicked = self.RC_Start
        )
        
        self.currentPort = QLabel()
        self.currentBaud = QLabel()
        
        self.canvas = pg.GraphicsLayoutWidget()             # create GrpahicsLayoutWidget obejct
        self.canvas2 = pg.GraphicsLayoutWidget()             # create GrpahicsLayoutWidget obejct

        self.SMPIN = QLabel("Sample interval (s): ")
        self.sampinterval = QLineEdit("0.001")
        self.TIMWIN = QLabel("Time window of plot (s): ")
        self.timewin = QLineEdit("20")

        sampleinterval = float(self.sampinterval.text())
        timewindow = float(self.timewin.text())
        self._interval = int(sampleinterval*1000)
        self._bufsize = int(timewindow/sampleinterval)
        self.databuffer1 = collections.deque([0.0]*self._bufsize, self._bufsize)
        self.databuffer2 = collections.deque([0.0]*self._bufsize, self._bufsize)
        self.databuffer3 = collections.deque([0.0]*self._bufsize, self._bufsize)
        self.x = np.linspace(-timewindow, 0.0, self._bufsize)
        self.y1 = np.zeros(self._bufsize, dtype=np.float64)
        self.y2 = np.zeros(self._bufsize, dtype=np.float64)
        self.y3 = np.zeros(self._bufsize, dtype=np.float64)
        self.AllData = []
        self.tstamp = []

        self.plt = self.canvas.addPlot(title='serial monitor')
        self.plt.resize(*(600,350))
        self.plt.showGrid(x=True, y=True)
        self.plt.setLabel('left', 'data', 'V')
        self.plt.setLabel('bottom', 'time', 's')
        self.curve1 = self.plt.plot(self.x, self.y1, pen=(0,255,255))
        self.curve2 = self.plt.plot(self.x, self.y2, pen=(255,0,255))
        self.curve3 = self.plt.plot(self.x, self.y3, pen=(255,255,0))

        self.plt2 = self.canvas2.addPlot(title='current')
        self.plt2.resize(*(600,350))
        self.plt2.showGrid(x=True, y=True)
        self.plt2.setLabel('left', 'amplitude', 'uA')
        self.plt2.setLabel('bottom', 'time', 's')
        self.curvep2l1 = self.plt2.plot(self.x, self.y1, pen=(0,255,255))

        # #cross hair
        # self.vLineP1 = pg.InfiniteLine(angle=90, movable=False)
        # self.hLineP1 = pg.InfiniteLine(angle=0, movable=False)
        # self.plt.addItem(self.vLineP1, ignoreBounds=True)
        # self.plt.addItem(self.hLineP1, ignoreBounds=True)
        # self.vbP1 = self.plt.vb

        self.plotNow = QPushButton(
            text = "Plot Now",
            checkable = True,
            clicked = self.updateplot
        )

        self.clearNow = QPushButton(
            text = "Clear Now",
            checkable = True,
            clicked = self.clearplot
        )

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateplot)
        self.timer.start(self._interval)

        

        self.saveImage = QPushButton(
            text = "Capture Image",
            checkable = True,
            toggled = self.save_image
        )

        self.saveCSV = QPushButton(
            text = "Save CSV",
            checkable = True,
            toggled = self.save_CSV
        )
        self.csv_counter = 1

        
        # set up image exporter (necessary to be able to export images)
        self.exporter=pg.exporters.ImageExporter(self.canvas.scene())
        self.image_counter = 1

        MLay = QHBoxLayout(self)

        lay = QVBoxLayout()
        hlay = QHBoxLayout()
        hlay.addWidget(self.command_line)
        hlay.addWidget(self.send_button)
        lay.addWidget(self.currentPort)
        lay.addWidget(self.currentBaud)
        lay.addWidget(self.configure_button)
        lay.addLayout(hlay)
        lay.addWidget(self.output_serial)
        lay.addWidget(self.connect_button)

        layD = QFormLayout(self)
        layD.addRow("Vds (mV) = ",self.RC_vdsInput)
        layD.addRow("Vgs (mV) = ",self.RC_pulseHInput)
        layD.addWidget(self.RC_send_btn)
        lay.addLayout(layD)
        
        Clay = QVBoxLayout()
        hlayWL = QHBoxLayout()
        hlayWL.addWidget(self.plotNow)
        hlayWL.addWidget(self.clearNow)
        hlayWL.addWidget(self.SMPIN)
        hlayWL.addWidget(self.sampinterval)
        hlayWL.addWidget(self.TIMWIN)
        hlayWL.addWidget(self.timewin)
        hlayWL.addWidget(self.saveImage)
        hlayWL.addWidget(self.saveCSV)
        Clay.addLayout(hlayWL)
        Clay.addWidget(self.canvas)
        Clay.addWidget(self.canvas2)
        
        MLay.addLayout(lay)
        MLay.addLayout(Clay)

        widget = QWidget()
        widget.setLayout(MLay)
        self.setCentralWidget(widget)
        self.serial = QtSerialPort.QSerialPort(self,readyRead=self.receive)

        # proxy1 = pg.SignalProxy(self.plt.scene().sigMouseMoved, rateLimit=60, slot=mousePoint)

    @QtCore.pyqtSlot()
    def COMport(self):
        config = COMportConfig()
        if config.exec_():
            portname, baudrate, baudno = config.get_results()
            self.serial.setPortName(portname)
            self.serial.setBaudRate(baudrate)
            self.currentPort.setText("Current port: " + portname)
            self.currentBaud.setText("Current baud rate: " + baudno)

    #@QtCore.pyqtSlot()
    def receive(self):
        while self.serial.canReadLine():
            text = self.serial.readLine().data().decode()
            text = text.rstrip('\r\n')
            textData = text.split(',')
            if len(textData)==3:                    
                text1 = textData[0]
                text2 = textData[1]
                text3 = textData[2]
                self.databuffer1.append(float(text1))
                self.databuffer2.append(float(text2))
                self.databuffer3.append(float(text3))
                self.AllData.append(list(map(float,textData)))
                self.tstamp.append(("%s" % time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())))
            else:
                pass
            self.output_serial.append(text)
            
            # self.output_serial.append(text)
            # self.databuffer.append(float(text))
            # self.AllData.append(float(text))
            # self.tstamp.append(("%s" % time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())))

    @QtCore.pyqtSlot()
    def send(self):
        self.serial.write(self.command_line.text().encode())
        print("sent.")

    #@QtCore.pyqtSlot()
    def RC_Start(self):
        vs = 1600
        vdscode = str(int((float(self.RC_vdsInput.text())+vs)/10))
        RCcommond1 = 'TransientVds ' + vdscode + ';'
        self.command_line.setText(RCcommond1)
        print(RCcommond1)
        self.send()
       
        # self.serial.write(RCcommond1.encode())
        while self.serial.canReadLine():
            text = self.serial.readLine().data().decode()
            text = text.rstrip('\r\n')
            self.output_serial.append(text)
        #timer/

        vgscode = str(int((float(self.RC_pulseHInput.text())+1000)/10))
        RCcommond2 = 'RCpulseinit ' + vgscode + ';'
        self.command_line.setText(RCcommond2)
        print(RCcommond2)
        self.send()
        # while self.serial.canReadLine():
        #     text = self.serial.readLine().data().decode()
        #     text = text.rstrip('\r\n')
        #     self.output_serial.append(text)

        # RCcommond3 = 'RCpulsetest 15;'
        # self.serial.write(RCcommond3.encode())


    @QtCore.pyqtSlot(bool)
    def on_toggled(self, checked):
        self.connect_button.setText("Disconnect" if checked else "Connect")
        if checked:
            if not self.serial.isOpen():
                self.serial.open(QtCore.QIODevice.ReadWrite)
                if not self.serial.isOpen():
                    self.connect_button.setChecked(False)
            else:
                self.connect_button.setChecked(False)
        else:
            self.serial.close()

    # @QtCore.pyqtSlot()
    def mouseMoved(self, e):
        print("move")
        mousePoint = self.vbP1.mapSceneToView(self.plt.scene().sigMouseMoved)
        index = int(mousePoint.x())
        # if index > 0 and index < len(data1):
        #     label.setText("<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (mousePoint.x(), data1[index], data2[index]))
        self.vLineP1.setPos(mousePoint.x())
        self.hLineP1.setPos(mousePoint.y())

    def updateplot(self):
        self.y1[:] = self.databuffer1
        self.y2[:] = self.databuffer2
        self.y3[:] = self.databuffer3
        self.curve1.setData(self.x, self.y1)
        self.curve2.setData(self.x, self.y2)
        self.curve3.setData(self.x, self.y3)

        self.curvep2l1.setData(self.x, self.y1)

    def clearplot(self):
        self.databuffer1 = collections.deque([0.0]*self._bufsize, self._bufsize)
        self.databuffer2 = collections.deque([0.0]*self._bufsize, self._bufsize)
        self.databuffer3 = collections.deque([0.0]*self._bufsize, self._bufsize)
        self.AllData = []
        self.tstamp = []
    
    def save_image(self, checked):
        self.saveImage.setText("Saving..." if checked else "Capture Image")
        if checked:
            filename = 'img'+("%04d" % self.image_counter)+("_%s" % time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime()))+'.png' 
            self.exporter.export(filename)
            self.image_counter += 1
            if os.path.exists(filename):
                self.saveImage.setChecked(False)
                print(filename)

    def save_CSV(self, checked):
        self.saveCSV.setText("Saving..." if checked else "Save CSV")
        if checked:
            cfilename = 'data'+("%04d" % self.csv_counter)+("_%s" % time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime()))+'.csv' 
            res = self.AllData.copy()
            timestamps = self.tstamp.copy()
            res = np.array(res)
            timestamps = np.array(timestamps)
            res = np.c_[timestamps,res]
            data = pd.DataFrame(data=res, columns=['timestamps'] + ['I'] + ['V1'] + ['V2'])
            data.to_csv(cfilename, float_format='%.3f', index=False)
            if os.path.exists(cfilename):
                self.saveCSV.setChecked(False)
                self.csv_counter += 1
                print(cfilename)
                self.AllData = []
                self.tstamp = []

    def closeEvent(self, event):
        self.serial.close()
        print("closed")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
