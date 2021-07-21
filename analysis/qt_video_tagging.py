from PyQt5.QtWidgets import QMainWindow, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QLabel, QLineEdit, QColorDialog, QComboBox, QSlider
from PyQt5.QtCore import QRect, QTimer, Qt
from PyQt5.QtGui import QColor, QPixmap, QImage
import cPickle as pickle
import os
from functools import partial

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import numpy as np
import cv_bridge

bridge = cv_bridge.CvBridge()

ROOT = '/home/main/data/2021_visual_servoing/active_perception'
ALL_FILES = sorted([f for f in os.listdir(ROOT) if f.endswith('.pickle')])

img_key = '/camera/color/image_raw/compressed'
wrench_key = '/wrench'

class DataLabeller(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle('Data labelling')

        # State variables
        self.data = None
        self.bounds = (0,0)
        self.last_fill_ref = None
        self.stamps = None
        self.stamps_to_indexes = {}
        self.processed_forces = None
        self.current_index = 0

        widget = QWidget()
        self.setCentralWidget(widget)
        main_layout = QVBoxLayout()
        widget.setLayout(main_layout)

        # Layout for image and graph
        display_widget = QWidget()
        display_layout = QHBoxLayout()
        display_widget.setLayout(display_layout)
        self.display_label = QLabel()
        display_layout.addWidget(self.display_label)

        fig = Figure((4, 3), dpi=100)
        self.axes = fig.add_subplot(111)
        self.plot_widget = FigureCanvasQTAgg(fig)
        self.lines = self.axes.plot([], np.zeros((0,3)))
        self.axes.legend(['X', 'Y', 'Z'], loc=3)
        self.axes.set_title('Force')
        display_layout.addWidget(self.plot_widget)

        main_layout.addWidget(display_widget)

        # Diagnostic label
        self.diagnostic_label = QLabel()
        main_layout.addWidget(self.diagnostic_label)

        # Layout for other stuff
        control_widget = QWidget()
        control_layout = QHBoxLayout()
        control_widget.setLayout(control_layout)

        self.slider_ticks = 1000
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.reset_slider()
        self.slider.valueChanged.connect(self.callback_slider)

        labels = ['No Contact', 'Contact', 'Failed', 'Success']
        self.label_buttons = [QPushButton(label) for label in labels]
        for i, button in enumerate(self.label_buttons):
            button.clicked.connect(partial(self.apply_label, i))
        clear_button = QPushButton('Clear')
        clear_button.clicked.connect(self.clear)
        save_button = QPushButton('Save')
        save_button.clicked.connect(self.save)


        control_layout.addWidget(self.slider)
        for label_button in self.label_buttons:
            control_layout.addWidget(label_button)
        control_layout.addWidget(clear_button)
        control_layout.addWidget(save_button)

        main_layout.addWidget(control_widget)

        # Layout for loading in file
        load_widget = QWidget()
        load_layout = QHBoxLayout()
        load_widget.setLayout(load_layout)
        start_text = ''
        if ALL_FILES:
            start_text = ALL_FILES[0]

        self.file_text = QLineEdit(start_text)
        load_button = QPushButton('Load')
        load_button.clicked.connect(self.load)
        next_button = QPushButton('>')
        next_button.clicked.connect(self.load_next)

        for widget in [self.file_text, load_button, next_button]:
            load_layout.addWidget(widget)

        main_layout.addWidget(load_widget)

    def load(self, check_index=True):
        file_name = self.file_text.text().strip()
        if check_index and file_name in ALL_FILES:
            self.current_index = ALL_FILES.index(file_name)

        with open(os.path.join(ROOT, file_name), 'rb') as fh:
            self.data = pickle.load(fh)
        if 'labels' not in self.data:
            self.data['labels'] = []
        else:
            print(self.data['labels'])

        if 'success' in self.data:
            print(self.data['success'])
            if self.data['success']:
                msg = 'File marked as a success'
            else:
                msg = 'File marked as a failure'
        else:
            msg = 'No success marking in file'
        self.diagnostic_label.setText(msg)

        stamps = sorted(self.data[img_key].keys())
        self.stamps = stamps
        self.stamps_to_indexes = {stamp: i for i, stamp in enumerate(self.stamps)}
        self.process_forces()

        start = self.data[img_key][stamps[0]].header.stamp.to_sec()
        end = self.data[img_key][stamps[-1]].header.stamp.to_sec()

        # Process the bounds if /int_state exists
        if '/int_state' in self.data:
            wiggling_ts = [ts for ts in self.data['/int_state'] if self.data['/int_state'][ts].data == 1]
            if not wiggling_ts:
                self.bounds = (0,0)
            else:
                self.bounds = (min(wiggling_ts), max(wiggling_ts))
        else:
            self.bounds = (0, 0)

        print(self.bounds)

        self.slider_ticks = min(1000, len(self.stamps))
        self.reset_slider()

    def load_next(self):

        if self.current_index + 1 == len(ALL_FILES):
            print('All files have been processed!')
            return
        self.current_index += 1
        self.file_text.setText(ALL_FILES[self.current_index])
        self.load(check_index=False)

    def save(self):
        file_name = self.file_text.text().strip()
        with open(os.path.join(ROOT, file_name), 'wb') as fh:
            pickle.dump(self.data, fh)
        print('File saved!')

    def clear(self):
        self.data['labels'] = []
        for button in self.label_buttons:
            button.setStyleSheet("background-color:#dddddd")
        print('Labels cleared')

    def reset_slider(self):
        self.slider.setMaximum(self.slider_ticks-1)
        self.slider.setValue(0)
        self.callback_slider()

    def callback_slider(self):
        if not self.stamps:
            return

        self.compressed_image_to_pixmap(self.data[img_key][self.current_stamp])
        self.recolor_buttons()
        self.redraw_plot()

    @property
    def current_stamp(self):
        current_val = self.slider.value()
        if self.slider_ticks != len(self.stamps) - 1:
            interp = float(current_val) / (self.slider_ticks - 1)
            current_val = int((len(self.stamps) - 1) * interp)
        stamp = self.stamps[current_val]
        return stamp

    def compressed_image_to_pixmap(self, img):
        img_array = bridge.compressed_imgmsg_to_cv2(img, desired_encoding='rgb8')
        h, w, c = img_array.shape
        q_image = QImage(img_array.data, w, h, 3*w, QImage.Format_RGB888)
        self.display_label.setPixmap(QPixmap(q_image))

    def apply_label(self, label):
        self.data['labels'].append((self.current_stamp, label))
        self.data['labels'] = sorted(self.data['labels'])
        print(self.data['labels'])
        self.callback_slider()

    def recolor_buttons(self):
        if not self.data['labels']:
            for button in self.label_buttons:
                button.setStyleSheet("background-color:#dddddd")
            return

        # Find the label of the current timestamp
        ts = self.current_stamp
        label = 0
        for next_ts, next_label in self.data['labels']:
            if next_ts > ts:
                break
            label = next_label

        # Iterate over each of the buttons and recolor them
        for i, button in enumerate(self.label_buttons):
            if i == label:
                button.setStyleSheet("background-color: green")
            else:
                button.setStyleSheet("background-color:#dddddd")


    def process_forces(self):
        # Goal: Associate a force value with each of the given timestamps
        wrench_data = self.data[wrench_key]
        wrench_stamps = sorted(wrench_data)
        force_array = np.array([[wrench_data[ts].wrench.force.x, wrench_data[ts].wrench.force.y, wrench_data[ts].wrench.force.z] for ts in wrench_stamps]).T

        interpolated_array = np.array([np.interp(self.stamps, wrench_stamps, force_array[i]) for i in range(3)])
        self.processed_forces = interpolated_array

        if not self.processed_forces.size:
            import pdb
            pdb.set_trace()

        self.axes.set_ylim(self.processed_forces.min(), self.processed_forces.max())

    def redraw_plot(self):

        current_idx = self.stamps_to_indexes[self.current_stamp]
        start_idx = max(0, current_idx - 30)

        t_data = np.array([self.stamps[start_idx:current_idx+1]]) - self.stamps[0]
        y_data = np.array(self.processed_forces[:,start_idx:current_idx+1])



        for line, data in zip(self.lines, y_data):
            line.set_xdata(t_data)
            line.set_ydata(data)

        start_time = self.stamps[start_idx] - self.stamps[0]
        end_time = max(self.stamps[current_idx] - self.stamps[0], start_time + 1)
        self.axes.set_xlim(start_time, end_time)


        # Draw recession bars for wiggling
        if self.last_fill_ref is not None:
            self.axes.collections.remove(self.last_fill_ref)

        bound_lower = self.bounds[0] - self.stamps[0]
        bound_upper = self.bounds[1] - self.stamps[0]
        if start_time <= bound_lower <= end_time or start_time <= bound_upper <= end_time or bound_lower <= start_time <= end_time <= bound_upper:
            fill_bound_lower = max(bound_lower, start_time)
            fill_bound_upper = min(bound_upper, end_time)
            y1, y2 = self.axes.get_ylim()
            self.last_fill_ref = self.axes.fill_between([fill_bound_lower, fill_bound_upper], y1=y1, y2=y2, alpha=0.25, color='green')
        else:
            self.last_fill_ref = None
        self.plot_widget.draw()


if __name__ == '__main__':
    app = QApplication([])
    gui = DataLabeller()
    gui.show()

    app.exec_()