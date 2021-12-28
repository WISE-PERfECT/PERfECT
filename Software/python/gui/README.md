# PERfECT GUI (pyqt)
Thanks to Ashira!

requirements:
pyqt5
pyqtgraph
numpy
pandas
[worked for me by pip install in a virtual environment]

Functions:
Connects to the user selected serial port (port + baud rate)
Plot the received data (3 data per line)
export png of plot (instantaneous)
CSV of data + timestamp can be saved; note: data matrix resets after saving csv
Command: single line, typed

To be implemented:
Preset commands
Change sampling window and interval size (currently the edits are not updated)
Further manipulation of data
[multithreading may be needed]
