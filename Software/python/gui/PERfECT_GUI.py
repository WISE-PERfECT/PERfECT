'''
Brief: The GUI for PERfECT system
Author: Xinyu Tian
Github: https://github.com/XITechs
CreateTime: 2022-05-02 18:33:27
LastEditTime: 2022-05-16 16:57:08
Description: 
'''

import sys
import numpy as np
from PyQt5 import QtCore, QtSerialPort
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QHeaderView, QFileDialog
from PyQt5.QtCore import QCoreApplication, QTimer
from PyQt5.QtSerialPort import QSerialPortInfo

from pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyqt5 import *
from pyqtgraph.graphicsItems.PlotItem.plotConfigTemplate_pyqt5 import *
from pyqtgraph.imageview.ImageViewTemplate_pyqt5 import *

import ion_bit

#from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import pyqtgraph.exporters as pgExporters
import collections
import os.path
import pandas as pd

import mainwindowui


#check version
# import inspect
# from PyQt5 import Qt
# vers = ['%s = %s' % (k,v) for k,v in vars(Qt).items() if k.lower().find('version') >= 0 and not inspect.isbuiltin(v)]
# print('\n'.join(sorted(vers)))

pg.setConfigOptions(antialias=True) #antialias
pg.setConfigOption('background', 'w') #w
pg.setConfigOption('foreground', 'k') #k
import ctypes
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid") #windows logo


#szValHpRtiaSe[] = { "200", "1k", "5k", "10k", "20k", "40k", "80k", "160k", "1Meg"}
RtiaList = [200, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 16000, 20000, 24000, 30000, 32000, 40000, 48000, 64000, 85000, 96000, 100000, 120000, 128000, 160000, 196000, 256000, 512000]

class Main(QMainWindow):
    def __init__(self):
         super().__init__()

         self.ui = mainwindowui.Ui_MainWindow()
         self.ui.setupUi(self)

         #setup ui--toolbar
         self.ui.actionConnect.triggered.connect(self.connectFunc)
         self.ui.actionStart.triggered.connect(self.startmeasFunc)
         self.ui.actionStop.triggered.connect(self.stopmeasFunc)
         self.ui.actionClear.triggered.connect(self.clearFunc)
         self.ui.actionCapture.triggered.connect(self.captureFunc)
         self.ui.actionSaveCSV.triggered.connect(self.saveCSVFunc)
         #setup ui--statusbar
         self.ui.statusbar.showMessage('Hi from PERfECT!             ----HKU WISE GROUP')

         #setup ui--dataview
         self.TransistorModeTableHead = ['Time (ms)', 'Vds (mV)', 'Vgs (mV)', 'Id (uA)']
         self.dataTableHeadNow = self.TransistorModeTableHead
         self.dataViewTableNum = self.ui.dataViewBufferSIzeSpinBox.value()
         self.ui.dataViewBufferSIzeSpinBox.valueChanged.connect(self.dataViewTableNumChangeFunc)  
         self.ui.dataViewTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
         self.ui.dataViewTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
         self.ui.dataViewTable.verticalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)

         # setup ui--plot control
         self.measStatusLEDFunc(0)
         self.connectStatusLEDFunc(0)


         #setup ui--plotgraph
         self.canvas = pg.GraphicsLayoutWidget()
         self.exporter=pgExporters.ImageExporter(self.canvas.scene())
         #self.plt = self.canvas.addPlot(row=0, col=0,title="<span style='font-size: 14pt'><span style='font-weight:bold'>PERfECT Plot</span>")
         self.plt = self.canvas.addPlot()
         self.plt.showGrid(x=True, y=True)
         self.plt.setLabel('left', 'Current', 'A')
         self.plt.setLabel('bottom', 'Index', '')
         self.plt.setMouseEnabled(x=True, y=False)
         self.plt.enableAutoRange(axis='y', enable=False)
         self.plt.enableAutoRange(axis='x', enable=True)
         self.curve1 = self.plt.plot([], [], pen=pg.mkPen('r', width=1))
         #self.curve2 = self.plt.plot(self.x, self.y2, pen=(255,0,255))
         #self.curve3 = self.plt.plot(self.x, self.y3, pen=(255,255,0))
         self.ui.plotLayout.addWidget(self.canvas)
         self.measTimeStampMS = 0

         #setup ui--config
         self.ui.comboBox.setCurrentIndex(0)
         self.meascfgFunc()
         self.ui.comboBox.currentIndexChanged.connect(self.meascfgFunc)

         #setup serial
         self.serial = QtSerialPort.QSerialPort(self)
         self.commandToSend = ""
         self.measFinishSignal = 0
         self.PERfECTport =0
         self.connectionStatusNow = 0

         #setup data buffer
         self._bufsize = self.ui.dataBufferSIzeSpinBox.value()
         self._bufsizeDisplay = self.ui.displayPointsNumSpinBox.value()
         self.ui.dataBufferSIzeSpinBox.valueChanged.connect(self.dataBufferSIzeSpinBoxFunc)
         self.ui.displayPointsNumSpinBox.valueChanged.connect(self.displayBufferChangeFunc)
         self.dataRecIndex = 0
         self.databuffer0 = collections.deque([ ]*self._bufsize, self._bufsize)
         self.databuffer1 = collections.deque([ ]*self._bufsize, self._bufsize)
         self.databuffer2 = collections.deque([ ]*self._bufsize, self._bufsize)
         self.databuffer3 = collections.deque([ ]*self._bufsize, self._bufsize)
         self.currentRtia  = RtiaList[0]
         #setup update timers
         self.timer = QTimer()
         self.timer.timeout.connect(self.dataUpdateFunc)
         self.timerSpeed = 50
         self.timer.start(self.timerSpeed)
         self.lastTimeIndex = 0

        
    def connectFunc(self,canConnect):
        canConnect = not self.serial.isOpen()
        self.ui.actionConnect.setText("Connect" if canConnect else "Disconnect")
        if canConnect:
            self.ui.actionConnect.setCheckable(False)
            print("-----------Connecting Start---------")
            for port in QSerialPortInfo.availablePorts():
                try:
                    print("Connecting to: " + port.portName())
                    self.serial.setPortName(port.portName())
                    self.serial.setBaudRate(230400)
                    if not self.serial.isOpen():
                        self.serial.open(QtCore.QIODevice.ReadWrite)
                        self.serial.setDataTerminalReady(True) 

                    if not self.serial.isOpen():    
                        canConnect=1
                        print('Cannot connect to: '+ port.portName())
                        self.serial.close()
                        #continue
                    else:
                        self.serial.write("connect;".encode())
                        if(self.serial.waitForReadyRead(2500)):
                            connectText = self.serial.readLine().data().decode()
                            connectText = connectText.rstrip('\r\n')
                            if(connectText=="322217"):
                                self.PERfECTport = port.portName()
                                print('Connected to: '+ self.PERfECTport)
                                self.ui.statusbar.showMessage('PERfECT connection built on ' + self.PERfECTport )
                                canConnect=0
                                self.connectionStatusNow = 1
                                self.connectStatusLEDFunc(self.connectionStatusNow)
                                self.measFinishSignal = 1
                                self.serial.readyRead.connect(self.receiveFunc)
                                self.PERfECTportInfo = port
                                self.ui.actionConnect.setText("Disconnect")
                                break
                            else:
                                canConnect=1
                                print('Cannot connect to: '+ port.portName())
                                self.serial.close()
                        else:
                            canConnect=1
                            print('Cannot connect to: '+ port.portName())
                            self.serial.close()
                except:
                    print("Errow with port: " + port.portName())
                    canConnect=1
                    self.serial.close()
                    #continue
            print("-----------Connecting End---------")
            if canConnect:
                self.ui.statusbar.showMessage('PERfECT connection failed, you can try again .')
        else:
            self.stopmeasFunc()
            while self.stopmeasFunc():pass
            canConnect=1
            self.serial.readyRead.disconnect(self.receiveFunc)
            self.serial.close()
            print("Disconnnect from port: " + self.serial.portName())
            self.ui.statusbar.showMessage('Disconnect PERfECT.')
            self.connectionStatusNow = 0
            self.connectStatusLEDFunc(self.connectionStatusNow)
            canConnect = not self.serial.isOpen()
            self.ui.actionConnect.setText("Connect" if canConnect else "Disconnect")
            self.ui.actionConnect.setCheckable(True)
        canConnect = not self.serial.isOpen()
        self.ui.actionConnect.setText("Connect" if canConnect else "Disconnect")
    
    def startmeasFunc(self):
        while(not self.CommandMaker()):pass
        if self.serial.isOpen():
            try:
                self.serial.write(self.commandToSend.encode())
                self.ui.statusbar.showMessage('Send command: ' + self.commandToSend)
                if(self.serial.waitForReadyRead(2500)):
                    connectText = self.serial.readLine().data().decode()
                    connectText = connectText.rstrip('\r\n')
                    if connectText=="~Start;" or connectText=="Start;":
                        self.measFinishSignal = 0
                        self.measStatusLEDFunc(not self.measFinishSignal)
                        return True
            except:
                self.ui.statusbar.showMessage('Bad connection!')
        else:
            self.ui.statusbar.showMessage('PERfECT not connected!')


    def stopmeasFunc(self):
        if self.serial.isOpen():
            try:
                self.serial.write("stop;".encode())
                self.ui.statusbar.showMessage('Send command: stop;')
                self.measFinishSignal = 0
                if(self.serial.waitForReadyRead(2500)):
                    connectText = self.serial.readLine().data().decode()
                    connectText = connectText.rstrip('\r\n')
                    if connectText=="~Finish;" or connectText=="Finish;":
                        self.measFinishSignal = 1
                        self.measStatusLEDFunc(not self.measFinishSignal)
                        return True
            except:
                self.ui.statusbar.showMessage('Bad connection!')
        else:
            self.ui.statusbar.showMessage('PERfECT not connected!')
    
    def measStatusLEDFunc(self,status):
        if status:
            self.ui.measStatusCheckBox.setStyleSheet("QCheckBox::indicator"
                                    "{"
                                    "background-color : lightgreen;" #lightgreen
                                    "}")
        else:
            self.ui.measStatusCheckBox.setStyleSheet("QCheckBox::indicator"
                                    "{"
                                    "background-color : gray;" #lightgreen
                                    "}")

    def connectStatusLEDFunc(self,status):
        if status:
            self.ui.connectionStatusCheckBox.setStyleSheet("QCheckBox::indicator"
                                    "{"
                                    "background-color : lightgreen;" #lightgreen
                                    "}")
        else:
            self.ui.connectionStatusCheckBox.setStyleSheet("QCheckBox::indicator"
                                    "{"
                                    "background-color : gray;" #lightgreen
                                    "}")

    def clearFunc(self):
        self.dataRecIndex = 0
        self.lastTimeIndex = 0
        self.databuffer0 = collections.deque([]*self._bufsize, self._bufsize)
        self.databuffer1 = collections.deque([]*self._bufsize, self._bufsize)
        self.databuffer2 = collections.deque([]*self._bufsize, self._bufsize)
        self.databuffer3 = collections.deque([]*self._bufsize, self._bufsize)
        self.curve1.setData([],[])
        
    
    def receiveFunc(self):
        textNonData = ""
        while self.serial.canReadLine():
            rawText = self.serial.readLine().data().decode()
            #print(text)
            text = rawText.rstrip('\r\n')
            textData = text.split(',')
            try:
                float(textData[0])
                if len(textData)==3:
                    text1 = float(textData[0])
                    text2 = float(textData[1])
                    text3 = float(textData[2])
                    self.databuffer0.append(self.dataRecIndex * self.measTimeStampMS)
                    self.databuffer1.append(self.adc2voltageFunc(text1))
                    self.databuffer2.append(self.adc2voltageFunc(text2))
                    self.databuffer3.append(self.adc2currentFunc(text3))
                    self.dataRecIndex = (self.dataRecIndex + 1)
                    self.measStatusLEDFunc(1)
                    

            except ValueError:
                if text=="~Finish;" or text=="Finish;":
                    self.measFinishSignal = 1
                    self.measStatusLEDFunc(not self.measFinishSignal)
                textNonData = textNonData + str(text)
                self.ui.statusbar.showMessage('Receive: '+textNonData)


    def dataBufferSIzeSpinBoxFunc(self):
        self._bufsize = self.ui.dataBufferSIzeSpinBox.value()
        databuffer0_back = self.databuffer0
        databuffer1_back = self.databuffer1
        databuffer2_back = self.databuffer2
        databuffer3_back = self.databuffer3
        self.databuffer0 = collections.deque([0.0]*self._bufsize, self._bufsize)
        self.databuffer1 = collections.deque([0.0]*self._bufsize, self._bufsize)
        self.databuffer2 = collections.deque([0.0]*self._bufsize, self._bufsize)
        self.databuffer3 = collections.deque([0.0]*self._bufsize, self._bufsize)
        self.databuffer0 = databuffer0_back
        self.databuffer1 = databuffer1_back
        self.databuffer2 = databuffer2_back
        self.databuffer3 = databuffer3_back
         
    def dataViewTableNumChangeFunc(self):
        self.dataViewTableNum = self.ui.dataViewBufferSIzeSpinBox.value()
        self.ui.dataViewTable.setRowCount(self.dataViewTableNum)

    def displayBufferChangeFunc(self):
        self._bufsizeDisplay = self.ui.displayPointsNumSpinBox.value()

    
    def plotPenControlFunction(self):
        self.dataTableHeadNow = self.TransistorModeTableHead
        characterizationNow = self.ui.comboBox.currentIndex()

        if(characterizationNow==0):
            self.plt.setTitle("PERfECT Plot")
            self.plt.setLabel('left', 'Current', 'A')
            self.plt.setLabel('bottom', 'Index', '')
        elif(characterizationNow==1 ):
             self.plt.setTitle("Transfer Curve")
             self.plt.setLabel('left', 'Current', 'A')
             self.plt.setLabel('bottom', 'Vgs', 'V')
        elif(characterizationNow==2 ):
            self.plt.setTitle("Output Curve")
            self.plt.setLabel('left', 'Current', 'A')
            self.plt.setLabel('bottom', 'Vds', 'V')
        elif(characterizationNow==3 ):
            self.plt.setTitle("Transient Responds")
            self.plt.setLabel('left', 'Current', 'A')
            self.plt.setLabel('bottom', 'Time', 's')
        elif(characterizationNow==4 ):
            self.plt.setTitle("I-T Curve")
            self.plt.setLabel('left', 'Current', 'A')
            self.plt.setLabel('bottom', 'Time', 's')
        self.ui.dataViewTable.setHorizontalHeaderLabels(self.dataTableHeadNow)
    
    def dataUpdateFunc(self):
        # if self.connectionStatusNow and not self.serial.aboutToClose():
        #     self.serial.readyRead.disconnect(self.receiveFunc)
        #     self.serial.close()
        #     print("Disconnnect from port: " + self.serial.portName())
        #     self.ui.statusbar.showMessage('Disconnect PERfECT.')
        #     self.connectStatusLEDFunc(0)
        #     canConnect = not self.serial.isOpen()
        #     self.ui.actionConnect.setText("Connect" if canConnect else "Disconnect")
        #     self.ui.actionConnect.setCheckable(True)

        # dataViewBrowserList0 = list(self.databuffer0)
        # dataViewBrowserListDisp0 = dataViewBrowserList0[-self.dataViewTableNum-1:]
        # dataViewBrowserList1 = list(self.databuffer1)
        # dataViewBrowserListDisp1 = dataViewBrowserList1[-self.dataViewTableNum-1:]
        # dataViewBrowserList2 = list(self.databuffer2)
        # dataViewBrowserListDisp2 = dataViewBrowserList2[-self.dataViewTableNum-1:]
        # dataViewBrowserList3 = list(self.databuffer3)
        # dataViewBrowserListDisp3 = dataViewBrowserList3[-self.dataViewTableNum-1:]
        dataViewBrowserList0 = np.array(self.databuffer0)
        dataViewBrowserListDisp0 = dataViewBrowserList0[-self.dataViewTableNum-1:]
        dataViewBrowserList1 = np.array(self.databuffer1)
        dataViewBrowserListDisp1 = dataViewBrowserList1[-self.dataViewTableNum-1:]
        dataViewBrowserList2 = np.array(self.databuffer2)
        dataViewBrowserListDisp2 = dataViewBrowserList2[-self.dataViewTableNum-1:]
        dataViewBrowserList3 = np.array(self.databuffer3)
        dataViewBrowserListDisp3 = dataViewBrowserList3[-self.dataViewTableNum-1:]
        
        dataPlotList0 = dataViewBrowserList0[-self._bufsizeDisplay:]/1000
        dataPlotList1 = dataViewBrowserList3[-self._bufsizeDisplay:]/1000000
        # try:
        # currentTimeIndex = dataPlotList0[-1]
        # newDataNum = currentTimeIndex - self.lastTimeIndex
        # if newDataNum:
        #     timeStampMS = (currentTimeIndex - self.lastTimeIndex)/self.timerSpeed
        #     self.lastTimeIndex = currentTimeIndex
        #     dataPlotList0 = dataPlotList0*timeStampMS
        #     print(newDataNum)
        # except:
        #     pass
        #self.plt.disableAutoRange()
        self.curve1.setData(dataPlotList0, dataPlotList1)
        # if dataPlotList0:
        #     self.plt.autoRange()

        try:
            for dataViewLineNumber in range(self.dataViewTableNum):
                dataViewItem0 = dataViewBrowserListDisp0[-dataViewLineNumber-1]
                #self.ui.dataViewTable.setItem(dataViewLineNumber,0,QTableWidgetItem(str(dataViewItem0)))
                self.ui.dataViewTable.setItem(dataViewLineNumber,0,QTableWidgetItem("%.3f" % dataViewItem0))
                dataViewItem1 = dataViewBrowserListDisp1[-dataViewLineNumber-1]
                self.ui.dataViewTable.setItem(dataViewLineNumber,1,QTableWidgetItem("%.3f" % dataViewItem1))
                dataViewItem2 = dataViewBrowserListDisp2[-dataViewLineNumber-1]
                self.ui.dataViewTable.setItem(dataViewLineNumber,2,QTableWidgetItem("%.3f" % dataViewItem2))
                dataViewItem3 = dataViewBrowserListDisp3[-dataViewLineNumber-1]
                self.ui.dataViewTable.setItem(dataViewLineNumber,3,QTableWidgetItem("%.3f" % dataViewItem3))
        except:
            pass
    
    
    def adc2voltageFunc(self,adcData): #mV
        return 1835 * ((adcData - 32768.0) / 32768.0) /1.5 + 32 #35mV for unknown adc error 
    def adc2currentFunc(self,adcData): #uA
        I = 1000*self.adc2voltageFunc(adcData) / self.currentRtia
        return I

    def captureFunc(self):
        fileName, ok = QFileDialog.getSaveFileName(self, "Save Capture", os.getcwd(), "PNG Files (*.png);;JPEG Files (*.jpg);;SVG Files (*.svg);")
        self.exporter.params['width'] = 1000
        self.exporter.export(fileName)

    def saveCSVFunc(self):
        fileName, ok = QFileDialog.getSaveFileName(self, "Save Data", os.getcwd(), "CSV Files (*.csv);")
        allDataInBuff = np.array([list(self.databuffer0), list(self.databuffer1), list(self.databuffer2), list(self.databuffer3)])
        allDataInBuff = allDataInBuff[:,-1-self.dataRecIndex:]
        dataCSV = pd.DataFrame(data=allDataInBuff.T, columns=self.dataTableHeadNow)
        dataCSV.to_csv(fileName, float_format='%.5f', index=False)
 

    def meascfgFunc(self):
        characterizationNow = self.ui.comboBox.currentIndex()
        if(characterizationNow==0):
            self.ui.MeasPara1Label.setText("")
            self.ui.measPara1Data.setVisible(False)
            self.ui.MeasUnit1Label.setText("")
            self.ui.MeasPara2Label.setText("")
            self.ui.measPara2Data.setVisible(False)
            self.ui.MeasUnit2Label.setText("")
            self.ui.MeasPara3Label.setText("")
            self.ui.measPara3Data.setVisible(False)
            self.ui.MeasUnit3Label.setText("")
            self.ui.MeasPara4Label.setText("")
            self.ui.measPara4Data.setVisible(False)
            self.ui.MeasUnit4Label.setText("")
            self.ui.MeasPara5Label.setText("")
            self.ui.measPara5Data.setVisible(False)
            self.ui.MeasUnit5Label.setText("")
            self.ui.MeasPara6Label.setText("")
            self.ui.measPara6Data.setVisible(False)
            self.ui.MeasUnit6Label.setText("")
            self.ui.MeasPara7Label.setText("")
            self.ui.measPara7Data.setVisible(False)
            self.ui.MeasUnit7Label.setText("")
            self.ui.MeasPara8Label.setText("")
            self.ui.measPara8Data.setVisible(False)
            self.ui.MeasUnit8Label.setText("")
            self.RtiaPosition = 0
        elif(characterizationNow==1): #transfer
            self.ui.MeasPara1Label.setText("Vgs Start")
            self.ui.measPara1Data.setDisabled(False)
            self.ui.measPara1Data.setVisible(True)
            self.ui.measPara1Data.setValue(-200)
            self.ui.MeasUnit1Label.setText("mV")
            self.ui.MeasPara2Label.setText("Vgs End")
            self.ui.measPara2Data.setDisabled(False)
            self.ui.measPara2Data.setVisible(True)
            self.ui.measPara2Data.setValue(200)
            self.ui.MeasUnit2Label.setText("mV")
            self.ui.MeasPara3Label.setText("Vgs Step")
            self.ui.measPara3Data.setDisabled(False)
            self.ui.measPara3Data.setVisible(True)
            self.ui.measPara3Data.setValue(50)
            self.ui.MeasUnit3Label.setText("mV")
            self.ui.MeasPara4Label.setText("Time Step")
            self.ui.measPara4Data.setDisabled(False)
            self.ui.measPara4Data.setVisible(True)
            self.ui.measPara4Data.setValue(100)
            self.ui.MeasUnit4Label.setText("ms")
            self.ui.MeasPara5Label.setText("Vds")
            self.ui.measPara5Data.setDisabled(False)
            self.ui.measPara5Data.setVisible(True)
            self.ui.measPara5Data.setValue(-400)
            self.ui.MeasUnit5Label.setText("mV")
            self.ui.MeasPara6Label.setText("Sensitivity")
            self.ui.measPara6Data.setDisabled(False)
            self.ui.measPara6Data.setVisible(True)
            self.ui.measPara6Data.setValue(1)
            self.ui.MeasUnit6Label.setText("")
            self.ui.MeasPara7Label.setText("Hysteresis")
            self.ui.measPara7Data.setDisabled(False)
            self.ui.measPara7Data.setVisible(True)
            self.ui.measPara7Data.setValue(1)
            self.ui.MeasUnit7Label.setText("")
            self.ui.MeasPara8Label.setText("")
            self.ui.measPara8Data.setDisabled(False)
            self.ui.measPara8Data.setVisible(False)
            self.ui.measPara8Data.setValue(1)
            self.ui.MeasUnit8Label.setText("")
        elif(characterizationNow==4): #transfer
            self.ui.MeasPara1Label.setText("Vgs")
            self.ui.measPara1Data.setDisabled(False)
            self.ui.measPara1Data.setVisible(True)
            self.ui.measPara1Data.setValue(200)
            self.ui.MeasUnit1Label.setText("mV")
            self.ui.MeasPara2Label.setText("Vds")
            self.ui.measPara2Data.setDisabled(False)
            self.ui.measPara2Data.setVisible(True)
            self.ui.measPara2Data.setValue(-400)
            self.ui.MeasUnit2Label.setText("mV")
            self.ui.MeasPara3Label.setText("Time step")
            self.ui.measPara3Data.setDisabled(False)
            self.ui.measPara3Data.setVisible(True)
            self.ui.measPara3Data.setValue(50)
            self.ui.MeasUnit3Label.setText("ms")
            self.ui.MeasPara4Label.setText("Time length")
            self.ui.measPara4Data.setDisabled(False)
            self.ui.measPara4Data.setVisible(True)
            self.ui.measPara4Data.setValue(10)
            self.ui.MeasUnit4Label.setText("s")
            self.ui.MeasPara5Label.setText("Sensitivity")
            self.ui.measPara5Data.setDisabled(False)
            self.ui.measPara5Data.setVisible(True)
            self.ui.measPara5Data.setValue(1)
            self.ui.MeasUnit5Label.setText("")
            self.ui.MeasPara6Label.setText("")
            self.ui.measPara6Data.setDisabled(False)
            self.ui.measPara6Data.setVisible(False)
            self.ui.measPara6Data.setValue(1)
            self.ui.MeasUnit6Label.setText("")
            self.ui.MeasPara7Label.setText("")
            self.ui.measPara7Data.setDisabled(False)
            self.ui.measPara7Data.setVisible(False)
            self.ui.measPara7Data.setValue(1)
            self.ui.MeasUnit7Label.setText("")
            self.ui.MeasPara8Label.setText("")
            self.ui.measPara8Data.setDisabled(False)
            self.ui.measPara8Data.setVisible(False)
            self.ui.measPara8Data.setValue(1)
            self.ui.MeasUnit8Label.setText("")
        self.plotPenControlFunction()


    def CommandMaker(self):
        characterizationNow = self.ui.comboBox.currentIndex()
        if characterizationNow == 0:
            fullcommand = "hi;"
            self.commandToSend = fullcommand
            return True
        elif characterizationNow == 1:
            fullcommand = self.commandStitcher("transconfig")
            self.commandToSend = fullcommand
            self.currentRtia = RtiaList[int(self.ui.measPara6Data.value())-1]
            return True
        elif characterizationNow == 3:
            fullcommand = "transientcfg %s;"% (",".join([self.config01content.text(),self.config02content.text(),self.config03content.text(),self.config04content.text(),self.config05content.text(),self.config06content.text(),self.config07content.text(),self.config08content.text(),self.config09content.text(),self.config10content.text(),self.config11content.text()]))
            self.commandToSend = fullcommand
            return True
        elif characterizationNow == 4:
            fullcommand = self.commandStitcher("caectcfg")
            self.commandToSend = fullcommand
            self.currentRtia = RtiaList[int(self.ui.measPara5Data.value())-1]
            self.measTimeStampMS = self.ui.measPara3Data.value()
            return True
        elif characterizationNow == 99:
            fullcommand = "washectcfg %s;"% (",".join([self.measPara1Data.d,self.config02content.text(),self.config03content.text(),self.config04content.text(),self.config05content.text(),self.config06content.text(),self.config07content.text()]))
            self.commandToSend = fullcommand
            return True
        
        elif characterizationNow == 98:
            fullcommand = "swvconfig %s;" % (",".join([self.config01content.text(),self.config02content.text(),self.config03content.text(),self.config04content.text(),self.config05content.text(),self.config06content.text(),self.config07content.text()]))
            self.commandToSend = fullcommand
            return True
    def commandStitcher(self,commandhead):
        return commandhead+(" %s;"% (",".join([str(self.ui.measPara1Data.value()),str(self.ui.measPara2Data.value()),str(self.ui.measPara3Data.value()),str(self.ui.measPara4Data.value()),str(self.ui.measPara5Data.value()),str(self.ui.measPara6Data.value()),str(self.ui.measPara7Data.value()),str(self.ui.measPara8Data.value())])))


if __name__ == '__main__':
    QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling) #self adjust resulotion
    app = QApplication(sys.argv)

    window = Main()
    window.setWindowIcon(QtGui.QIcon(':/appIcon.ico'))
    window.showMaximized()
    #window.show()
    sys.exit(app.exec_())
