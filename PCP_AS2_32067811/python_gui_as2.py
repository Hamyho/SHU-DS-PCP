import sys
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.constants import *
import os.path

from eda_module import EDA
import pandas as pd
from io import StringIO
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

_script = sys.argv[0]
_location = os.path.dirname(_script)
_source_data = pd.read_csv('activity_context_tracking_data.csv').drop('_id', axis=1)

import python_gui_as2_support

_bgcolor = '#d9d9d9'  # X11 color: 'gray85'
_fgcolor = '#000000'  # X11 color: 'black'
_compcolor = 'gray40'  # X11 color: #666666
_ana1color = '#c3c3c3'  # Closest X11 color: 'gray76'
_ana2color = 'beige'  # X11 color: #f5f5dc
_tabfg1 = 'black'
_tabfg2 = 'black'
_tabbg1 = 'grey75'
_tabbg2 = 'grey89'
_bgmode = 'light'

_style_code_ran = 0


def _style_code():
    global _style_code_ran
    if _style_code_ran:
        return
    style = ttk.Style()
    if sys.platform == "win32":
        style.theme_use('winnative')
    style.configure('.', background=_bgcolor)
    style.configure('.', foreground=_fgcolor)
    style.configure('.', font='TkDefaultFont')
    style.map('.', background=
    [('selected', _compcolor), ('active', _ana2color)])
    if _bgmode == 'dark':
        style.map('.', foreground=
        [('selected', 'white'), ('active', 'white')])
    else:
        style.map('.', foreground=
        [('selected', 'black'), ('active', 'black')])
    style.map('TNotebook.Tab', background=
    [('selected', _bgcolor), ('active', _tabbg1),
     ('!active', _ana2color)], foreground=
              [('selected', _fgcolor), ('active', _tabfg1), ('!active', _tabfg2)])
    style.configure('Vertical.TScrollbar', background=_bgcolor,
                    arrowcolor=_fgcolor)
    style.configure('Horizontal.TScrollbar', background=_bgcolor,
                    arrowcolor=_fgcolor)
    style.configure('Treeview', font="TkDefaultFont")
    _style_code_ran = 1


class UserInterface:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''

        top.geometry("1096x711+485+44")
        top.minsize(1, 1)
        top.maxsize(1905, 1050)
        top.resizable(1, 1)
        top.title("Machine Learning - Activities")
        top.configure(highlightcolor="black")

        self.top = top
        self.checked_orX = tk.IntVar()
        self.checked_orY = tk.IntVar()
        self.checked_orZ = tk.IntVar()
        self.checked_rX = tk.IntVar()
        self.checked_rY = tk.IntVar()
        self.checked_rZ = tk.IntVar()
        self.checked_accX = tk.IntVar()
        self.checked_accY = tk.IntVar()
        self.checked_accZ = tk.IntVar()
        self.checked_mX = tk.IntVar()
        self.checked_mY = tk.IntVar()
        self.checked_mZ = tk.IntVar()
        self.checked_lux = tk.IntVar()
        self.checked_gX = tk.IntVar()
        self.checked_gY = tk.IntVar()
        self.checked_gZ = tk.IntVar()
        self.checked_sound_level = tk.IntVar()
        self.checked_SVM = tk.IntVar()
        self.checked_random_forest = tk.IntVar()
        self.checked_mlp = tk.IntVar()
        self.checked_knn = tk.IntVar()
        self.checked_decision_tree = tk.IntVar()
        self.checked_logistic = tk.IntVar()
        self.checked_op_orX = tk.IntVar()
        self.checked_op_orY = tk.IntVar()
        self.checked_op_orZ = tk.IntVar()
        self.checked_op_rX = tk.IntVar()
        self.checked_op_rY = tk.IntVar()
        self.checked_op_rZ = tk.IntVar()
        self.checked_op_accX = tk.IntVar()
        self.checked_op_accY = tk.IntVar()
        self.checked_op_accZ = tk.IntVar()
        self.checked_op_mX = tk.IntVar()
        self.checked_op_mY = tk.IntVar()
        self.checked_op_mZ = tk.IntVar()
        self.checked_op_lux = tk.IntVar()
        self.checked_op_gX = tk.IntVar()
        self.checked_op_gY = tk.IntVar()
        self.checked_op_gZ = tk.IntVar()
        self.checkeck_op_soundlevel = tk.IntVar()

        self.user = EDA(_source_data)
        self.source_data = self.user.source_data
        self.y_predicts = {}

        self.menubar = tk.Menu(top, font="TkMenuFont", bg=_bgcolor, fg=_fgcolor)
        top.configure(menu=self.menubar)

        _style_code()
        self.TNotebook1 = ttk.Notebook(self.top)
        self.TNotebook1.place(relx=0.0, rely=0.0, relheight=1.001
                              , relwidth=1.001)
        self.TNotebook1.configure(takefocus="")
        self.tab_retrieve = tk.Frame(self.TNotebook1)
        self.TNotebook1.add(self.tab_retrieve, padding=3)
        self.TNotebook1.tab(0, text='''Retrieve''', compound="left"
                            , underline='''-1''', )
        self.tab_statistic = tk.Frame(self.TNotebook1)
        self.TNotebook1.add(self.tab_statistic, padding=3)
        self.TNotebook1.tab(1, text='''Statistic''', compound="left"
                            , underline='''-1''', )
        self.tab_freq_depend = tk.Frame(self.TNotebook1)
        self.TNotebook1.add(self.tab_freq_depend, padding=3)
        self.TNotebook1.tab(2, text='''Frequencies/Dependencies''', compound="left"
                            , underline='''-1''', )
        self.tab_freq_depend.bind("<Visibility>", lambda event: self.run_frequency_dependence())
        self.tab_preprocessing = tk.Frame(self.TNotebook1)
        self.TNotebook1.add(self.tab_preprocessing, padding=3)
        self.TNotebook1.tab(3, text='''Preprocessing''', compound="left"
                            , underline='''-1''', )
        self.tab_train_predict = tk.Frame(self.TNotebook1)
        self.TNotebook1.add(self.tab_train_predict, padding=3)
        self.TNotebook1.tab(4, text='''Train/Predict''', compound="left"
                            , underline='''-1''', )
        self.tab_report = tk.Frame(self.TNotebook1)
        self.TNotebook1.add(self.tab_report, padding=3)
        self.TNotebook1.tab(5, text='''Report''', compound="left"
                            , underline='''-1''', )
        self.tab_report.bind("<Visibility>", lambda event: self.run_report())
        self.tab_confussion_matrix = tk.Frame(self.TNotebook1)
        self.TNotebook1.add(self.tab_confussion_matrix, padding=3)
        self.TNotebook1.tab(6, text='''Confusion Matrix''', compound="left"
                            , underline='''-1''', )
        self.tab_confussion_matrix.bind("<Visibility>", lambda event: self.run_confusion_matrix())
        self.tab_optional = tk.Frame(self.TNotebook1)
        self.TNotebook1.add(self.tab_optional, padding=3)
        self.TNotebook1.tab(7, text='''Optional Analysis''', compound="left"
                            , underline='''-1''', )
        # self.extra_tab1 = tk.Frame(self.TNotebook1)
        # self.TNotebook1.add(self.extra_tab1, padding=3)
        # self.TNotebook1.tab(8, text='''Extra Plot''', compound="left"
        #                     , underline='''-1''', )

        # self.TNotebook1_t10 = tk.Frame(self.TNotebook1)
        # self.TNotebook1.add(self.TNotebook1_t10, padding=3)
        # self.TNotebook1.tab(9, text='''New Tab''', compound="left"
        #                     , underline='''-1''', )
        # self.TNotebook1_t11 = tk.Frame(self.TNotebook1)
        # self.TNotebook1.add(self.TNotebook1_t11, padding=3)
        # self.TNotebook1.tab(10, text='''New Tab''', compound="left"
        #                     , underline='''-1''', )
        # self.TNotebook1_t13 = tk.Frame(self.TNotebook1)
        # self.TNotebook1.add(self.TNotebook1_t13, padding=3)
        # self.TNotebook1.tab(11, text='''New Tab''', compound="left"
        #                     , underline='''-1''', )
        self.frame_table_retrieve = ScrolledTreeView(self.tab_retrieve)
        self.frame_table_retrieve.place(relx=0.009, rely=0.03, relheight=0.941
                                        , relwidth=0.982)
        self.frame_table_retrieve.configure(columns=list(self.source_data.columns))
        # build_treeview_support starting.
        self.frame_table_retrieve.heading("#0", text="ID")
        self.frame_table_retrieve.heading("#0", anchor="center")
        self.frame_table_retrieve.column("#0", width="30")
        self.frame_table_retrieve.column("#0", minwidth="20")
        self.frame_table_retrieve.column("#0", stretch="1")
        self.frame_table_retrieve.column("#0", anchor="w")

        for column in self.source_data.columns:
            self.frame_table_retrieve.heading(column, text=column)
            self.frame_table_retrieve.heading(column, anchor="center")
            self.frame_table_retrieve.column(column, width="30")
            self.frame_table_retrieve.column(column, minwidth="20")
            self.frame_table_retrieve.column(column, stretch="1")
            self.frame_table_retrieve.column(column, anchor="w")
        # Remember remove command these code before submit
        # for index, row in self.source_data.iterrows():
        #     item_id = str(index)
        #     values = tuple(row[column] for column in self.source_data.columns)
        #     self.frame_table_retrieve.insert("", "end", text=item_id, values=values)

        self.features = tk.LabelFrame(self.tab_statistic)
        self.features.place(relx=0.018, rely=0.015, relheight=0.114
                            , relwidth=0.961)
        self.features.configure(relief='groove')
        self.features.configure(text='''Feature Interest''')
        self.orX = tk.Checkbutton(self.features)
        self.orX.place(relx=0.01, rely=0.403, relheight=0.299, relwidth=0.064
                       , bordermode='ignore')
        self.orX.configure(activebackground="beige")
        self.orX.configure(anchor='w')
        self.orX.configure(compound='left')
        self.orX.configure(justify='left')
        self.orX.configure(selectcolor="#d9d9d9")
        self.orX.configure(text='''orX''')
        self.orX.configure(variable=self.checked_orX)
        self.orY = tk.Checkbutton(self.features)
        self.orY.place(relx=0.067, rely=0.403, relheight=0.299, relwidth=0.065
                       , bordermode='ignore')
        self.orY.configure(activebackground="beige")
        self.orY.configure(anchor='w')
        self.orY.configure(compound='left')
        self.orY.configure(justify='left')
        self.orY.configure(selectcolor="#d9d9d9")
        self.orY.configure(text='''orY''')
        self.orY.configure(variable=self.checked_orY)
        self.orZ = tk.Checkbutton(self.features)
        self.orZ.place(relx=0.124, rely=0.403, relheight=0.299, relwidth=0.065
                       , bordermode='ignore')
        self.orZ.configure(activebackground="beige")
        self.orZ.configure(anchor='w')
        self.orZ.configure(compound='left')
        self.orZ.configure(justify='left')
        self.orZ.configure(selectcolor="#d9d9d9")
        self.orZ.configure(text='''orZ''')
        self.orZ.configure(variable=self.checked_orZ)
        self.rX = tk.Checkbutton(self.features)
        self.rX.place(relx=0.191, rely=0.403, relheight=0.299, relwidth=0.064
                      , bordermode='ignore')
        self.rX.configure(activebackground="beige")
        self.rX.configure(anchor='w')
        self.rX.configure(compound='left')
        self.rX.configure(justify='left')
        self.rX.configure(selectcolor="#d9d9d9")
        self.rX.configure(text='''rX''')
        self.rX.configure(variable=self.checked_rX)
        self.rY = tk.Checkbutton(self.features)
        self.rY.place(relx=0.239, rely=0.403, relheight=0.299, relwidth=0.065
                      , bordermode='ignore')
        self.rY.configure(activebackground="beige")
        self.rY.configure(anchor='w')
        self.rY.configure(compound='left')
        self.rY.configure(justify='left')
        self.rY.configure(selectcolor="#d9d9d9")
        self.rY.configure(text='''rY''')
        self.rY.configure(variable=self.checked_rY)
        self.rZ = tk.Checkbutton(self.features)
        self.rZ.place(relx=0.287, rely=0.403, relheight=0.299, relwidth=0.064
                      , bordermode='ignore')
        self.rZ.configure(activebackground="beige")
        self.rZ.configure(anchor='w')
        self.rZ.configure(compound='left')
        self.rZ.configure(justify='left')
        self.rZ.configure(selectcolor="#d9d9d9")
        self.rZ.configure(text='''rZ''')
        self.rZ.configure(variable=self.checked_rZ)
        self.accX = tk.Checkbutton(self.features)
        self.accX.place(relx=0.344, rely=0.403, relheight=0.299, relwidth=0.064
                        , bordermode='ignore')
        self.accX.configure(activebackground="beige")
        self.accX.configure(anchor='w')
        self.accX.configure(compound='left')
        self.accX.configure(justify='left')
        self.accX.configure(selectcolor="#d9d9d9")
        self.accX.configure(text='''accX''')
        self.accX.configure(variable=self.checked_accX)
        self.accY = tk.Checkbutton(self.features)
        self.accY.place(relx=0.401, rely=0.403, relheight=0.299, relwidth=0.064
                        , bordermode='ignore')
        self.accY.configure(activebackground="beige")
        self.accY.configure(anchor='w')
        self.accY.configure(compound='left')
        self.accY.configure(justify='left')
        self.accY.configure(selectcolor="#d9d9d9")
        self.accY.configure(text='''accY''')
        self.accY.configure(variable=self.checked_accY)
        self.accZ = tk.Checkbutton(self.features)
        self.accZ.place(relx=0.458, rely=0.403, relheight=0.299, relwidth=0.064
                        , bordermode='ignore')
        self.accZ.configure(activebackground="beige")
        self.accZ.configure(anchor='w')
        self.accZ.configure(compound='left')
        self.accZ.configure(justify='left')
        self.accZ.configure(selectcolor="#d9d9d9")
        self.accZ.configure(text='''accZ''')
        self.accZ.configure(variable=self.checked_accZ)
        self.mX = tk.Checkbutton(self.features)
        self.mX.place(relx=0.705, rely=0.403, relheight=0.299, relwidth=0.065
                      , bordermode='ignore')
        self.mX.configure(activebackground="beige")
        self.mX.configure(anchor='w')
        self.mX.configure(compound='left')
        self.mX.configure(justify='left')
        self.mX.configure(selectcolor="#d9d9d9")
        self.mX.configure(text='''mX''')
        self.mX.configure(variable=self.checked_mX)
        self.mY = tk.Checkbutton(self.features)
        self.mY.place(relx=0.753, rely=0.403, relheight=0.299, relwidth=0.064
                      , bordermode='ignore')
        self.mY.configure(activebackground="beige")
        self.mY.configure(anchor='w')
        self.mY.configure(compound='left')
        self.mY.configure(justify='left')
        self.mY.configure(selectcolor="#d9d9d9")
        self.mY.configure(text='''mY''')
        self.mY.configure(variable=self.checked_mY)
        self.mZ = tk.Checkbutton(self.features)
        self.mZ.place(relx=0.801, rely=0.403, relheight=0.299, relwidth=0.064
                      , bordermode='ignore')
        self.mZ.configure(activebackground="beige")
        self.mZ.configure(anchor='w')
        self.mZ.configure(compound='left')
        self.mZ.configure(justify='left')
        self.mZ.configure(selectcolor="#d9d9d9")
        self.mZ.configure(text='''mZ''')
        self.mZ.configure(variable=self.checked_mZ)
        self.lux = tk.Checkbutton(self.features)
        self.lux.place(relx=0.848, rely=0.403, relheight=0.299, relwidth=0.064
                       , bordermode='ignore')
        self.lux.configure(activebackground="beige")
        self.lux.configure(anchor='w')
        self.lux.configure(compound='left')
        self.lux.configure(justify='left')
        self.lux.configure(selectcolor="#d9d9d9")
        self.lux.configure(text='''lux''')
        self.lux.configure(variable=self.checked_lux)
        self.gX = tk.Checkbutton(self.features)
        self.gX.place(relx=0.544, rely=0.403, relheight=0.299, relwidth=0.065
                      , bordermode='ignore')
        self.gX.configure(activebackground="beige")
        self.gX.configure(anchor='w')
        self.gX.configure(compound='left')
        self.gX.configure(justify='left')
        self.gX.configure(selectcolor="#d9d9d9")
        self.gX.configure(text='''gX''')
        self.gX.configure(variable=self.checked_gX)
        self.gY = tk.Checkbutton(self.features)
        self.gY.place(relx=0.601, rely=0.403, relheight=0.299, relwidth=0.065
                      , bordermode='ignore')
        self.gY.configure(activebackground="beige")
        self.gY.configure(anchor='w')
        self.gY.configure(compound='left')
        self.gY.configure(justify='left')
        self.gY.configure(selectcolor="#d9d9d9")
        self.gY.configure(text='''gY''')
        self.gY.configure(variable=self.checked_gY)
        self.gZ = tk.Checkbutton(self.features)
        self.gZ.place(relx=0.656, rely=0.39, relheight=0.312, relwidth=0.064
                      , bordermode='ignore')
        self.gZ.configure(activebackground="beige")
        self.gZ.configure(anchor='w')
        self.gZ.configure(compound='left')
        self.gZ.configure(justify='left')
        self.gZ.configure(selectcolor="#d9d9d9")
        self.gZ.configure(text='''gZ''')
        self.gZ.configure(variable=self.checked_gZ)
        self.sound_level = tk.Checkbutton(self.features)
        self.sound_level.place(relx=0.896, rely=0.403, relheight=0.299
                               , relwidth=0.102, bordermode='ignore')
        self.sound_level.configure(activebackground="beige")
        self.sound_level.configure(anchor='w')
        self.sound_level.configure(compound='left')
        self.sound_level.configure(justify='left')
        self.sound_level.configure(selectcolor="#d9d9d9")
        self.sound_level.configure(text='''soundLevel''')
        self.sound_level.configure(variable=self.checked_sound_level)

        self.button_describe = tk.Button(self.tab_statistic)
        self.button_describe.place(relx=0.459, rely=0.136, height=33, width=73)
        self.button_describe.configure(activebackground="beige")
        self.button_describe.configure(borderwidth="2")
        self.button_describe.configure(command=self.run_describe)
        self.button_describe.configure(compound='left')
        self.button_describe.configure(text='''Describe''')
        self.name_statistic = ['Mean', 'Standard Deviation', 'Median', 'Variance', 'Minimum', 'Maximum', 'Skewness',
                               'Kurtosis']
        self.frame_statistic_table = ScrolledTreeView(self.tab_statistic)
        self.frame_statistic_table.place(relx=0.018, rely=0.196, relheight=0.774
                                         , relwidth=0.972)
        self.frame_statistic_table.configure(columns=self.name_statistic)
        # build_treeview_support starting.
        self.frame_statistic_table.heading("#0", text="Variables")
        self.frame_statistic_table.heading("#0", anchor="center")
        self.frame_statistic_table.column("#0", width=20)
        self.frame_statistic_table.column("#0", minwidth=20)
        self.frame_statistic_table.column("#0", stretch="1")
        self.frame_statistic_table.column("#0", anchor="w")

        for column in self.name_statistic:
            self.frame_statistic_table.heading(column, text=column)
            self.frame_statistic_table.heading(column, anchor="center")
            self.frame_statistic_table.column(column, width=20)
            self.frame_statistic_table.column(column, minwidth=20)
            self.frame_statistic_table.column(column, stretch="1")
            self.frame_statistic_table.column(column, anchor="w")

        self.frame_frequency = tk.LabelFrame(self.tab_freq_depend)
        self.frame_frequency.place(relx=0.018, rely=0.136, relheight=0.759
                                   , relwidth=0.468)
        self.frame_frequency.configure(relief='groove')
        self.frame_frequency.configure(text='''Frequencies''')
        self.frame_frequency.configure(background="#a6b9d8")
        self.frame_dependances = tk.LabelFrame(self.tab_freq_depend)
        self.frame_dependances.place(relx=0.505, rely=0.136, relheight=0.759
                                     , relwidth=0.478)
        self.frame_dependances.configure(relief='groove')
        self.frame_dependances.configure(text='''Dependences''')
        self.frame_dependances.configure(background="#adcad8")
        self.frame_data_info = tk.LabelFrame(self.tab_preprocessing)
        self.frame_data_info.place(relx=0.028, rely=0.044, relheight=0.901
                                   , relwidth=0.468)
        self.frame_data_info.configure(relief='groove')
        self.frame_data_info.configure(text='''Data Information''')
        self.info_text = ScrolledText(self.frame_data_info)
        self.info_text.place(relx=0.0, rely=0.032, relheight=0.958, relwidth=1.0
                             , bordermode='ignore')
        self.info_text.configure(background="white")
        self.info_text.configure(font="TkTextFont")
        self.info_text.configure(insertborderwidth="3")
        self.info_text.configure(selectbackground="#c4c4c4")
        self.info_text.configure(wrap="none")
        info_output = StringIO()
        self.source_data.info(buf=info_output)
        info_string = info_output.getvalue()
        self.info_text.insert("1.0", info_string)
        self.info_text.configure(state="disabled")

        self.button_clean = tk.Button(self.tab_preprocessing)
        self.button_clean.place(relx=0.532, rely=0.439, height=33, width=93)
        self.button_clean.configure(activebackground="beige")
        self.button_clean.configure(borderwidth="2")
        self.button_clean.configure(command=self.run_clean_data)
        self.button_clean.configure(compound='left')
        self.button_clean.configure(text='''Clean Data''')
        self.label_clean_completed = tk.Label(self.tab_preprocessing)
        self.label_clean_completed.place(relx=0.688, rely=0.455, height=22
                                         , width=160)
        self.label_clean_completed.configure(activebackground="#f9f9f9")
        self.label_clean_completed.configure(anchor='w')
        self.label_clean_completed.configure(compound='left')
        self.label_clean_completed.configure(text='''''')
        self.frame_models = tk.LabelFrame(self.tab_train_predict)
        self.frame_models.place(relx=0.018, rely=0.015, relheight=0.114
                                , relwidth=0.972)
        self.frame_models.configure(relief='groove')
        self.frame_models.configure(text='''Models''')
        self.SVM = tk.Checkbutton(self.frame_models)
        self.SVM.place(relx=0.066, rely=0.403, relheight=0.299, relwidth=0.063
                       , bordermode='ignore')
        self.SVM.configure(activebackground="beige")
        self.SVM.configure(anchor='w')
        self.SVM.configure(compound='left')
        self.SVM.configure(justify='left')
        self.SVM.configure(selectcolor="#d9d9d9")
        self.SVM.configure(text='''SVM''')
        self.SVM.configure(variable=self.checked_SVM)
        self.random_forest = tk.Checkbutton(self.frame_models)
        self.random_forest.place(relx=0.227, rely=0.403, relheight=0.299
                                 , relwidth=0.129, bordermode='ignore')
        self.random_forest.configure(activebackground="beige")
        self.random_forest.configure(anchor='w')
        self.random_forest.configure(compound='left')
        self.random_forest.configure(justify='left')
        self.random_forest.configure(selectcolor="#d9d9d9")
        self.random_forest.configure(text='''RandomForest''')
        self.random_forest.configure(variable=self.checked_random_forest)
        self.mlp = tk.Checkbutton(self.frame_models)
        self.mlp.place(relx=0.387, rely=0.403, relheight=0.299, relwidth=0.063
                       , bordermode='ignore')
        self.mlp.configure(activebackground="beige")
        self.mlp.configure(anchor='w')
        self.mlp.configure(compound='left')
        self.mlp.configure(justify='left')
        self.mlp.configure(selectcolor="#d9d9d9")
        self.mlp.configure(text='''MLP''')
        self.mlp.configure(variable=self.checked_mlp)
        self.KNN = tk.Checkbutton(self.frame_models)
        self.KNN.place(relx=0.548, rely=0.403, relheight=0.299, relwidth=0.063
                       , bordermode='ignore')
        self.KNN.configure(activebackground="beige")
        self.KNN.configure(anchor='w')
        self.KNN.configure(compound='left')
        self.KNN.configure(justify='left')
        self.KNN.configure(selectcolor="#d9d9d9")
        self.KNN.configure(text='''KNN''')
        self.KNN.configure(variable=self.checked_knn)
        self.decision_tree = tk.Checkbutton(self.frame_models)
        self.decision_tree.place(relx=0.689, rely=0.403, relheight=0.299
                                 , relwidth=0.11, bordermode='ignore')
        self.decision_tree.configure(activebackground="beige")
        self.decision_tree.configure(anchor='w')
        self.decision_tree.configure(compound='left')
        self.decision_tree.configure(justify='left')
        self.decision_tree.configure(selectcolor="#d9d9d9")
        self.decision_tree.configure(text='''DecisionTree''')
        self.decision_tree.configure(variable=self.checked_decision_tree)
        self.logistic = tk.Checkbutton(self.frame_models)
        self.logistic.place(relx=0.831, rely=0.403, relheight=0.299
                            , relwidth=0.092, bordermode='ignore')
        self.logistic.configure(activebackground="beige")
        self.logistic.configure(anchor='w')
        self.logistic.configure(compound='left')
        self.logistic.configure(justify='left')
        self.logistic.configure(selectcolor="#d9d9d9")
        self.logistic.configure(text='''Logistic''')
        self.logistic.configure(variable=self.checked_logistic)
        self.button_train = tk.Button(self.tab_train_predict)
        self.button_train.place(relx=0.367, rely=0.136, height=33, width=73)
        self.button_train.configure(activebackground="beige")
        self.button_train.configure(borderwidth="2")
        self.button_train.configure(command=self.run_train)
        self.button_train.configure(compound='left')
        self.button_train.configure(text='''Train''')
        self.button_predict = tk.Button(self.tab_train_predict)
        self.button_predict.place(relx=0.56, rely=0.136, height=33, width=73)
        self.button_predict.configure(activebackground="beige")
        self.button_predict.configure(borderwidth="2")
        self.button_predict.configure(command=self.run_predict)
        self.button_predict.configure(compound='left')
        self.button_predict.configure(text='''Predict''')
        self.frame_best_param = tk.LabelFrame(self.tab_train_predict)
        self.frame_best_param.place(relx=0.018, rely=0.258, relheight=0.706
                                    , relwidth=0.972)
        self.frame_best_param.configure(relief='groove')
        self.frame_best_param.configure(text='''Best Params''')
        self.frame_text_best_params = ScrolledText(self.frame_best_param)
        self.frame_text_best_params.place(relx=0.0, rely=0.042, relheight=0.946
                                          , relwidth=0.993, bordermode='ignore')
        self.frame_text_best_params.configure(background="white")
        self.frame_text_best_params.configure(font="TkTextFont")
        self.frame_text_best_params.configure(insertborderwidth="3")
        self.frame_text_best_params.configure(selectbackground="#c4c4c4")
        self.frame_text_best_params.configure(wrap="none")

        self.model_status = tk.Label(self.tab_train_predict)
        self.model_status.place(relx=0.457, rely=0.19, height=22, width=110)
        self.model_status.configure(anchor='w')
        self.model_status.configure(compound='left')
        self.model_status.configure(cursor="fleur")
        self.model_status.configure(text='''''')

        self.frame_rp_SVM = tk.LabelFrame(self.tab_report)
        self.frame_rp_SVM.place(relx=0.0, rely=0.0, relheight=0.508
                                , relwidth=0.341)
        self.frame_rp_SVM.configure(relief='groove')
        self.frame_rp_SVM.configure(text='''SVM''')
        self.tb_rp_svm = ScrolledTreeView(self.frame_rp_SVM)

        self.tb_rp_svm.place(relx=0.0, rely=0.057, relheight=0.951
                             , relwidth=1.126, bordermode='ignore')

        # build_treeview_support starting.
        self.tb_rp_svm.heading("#0", text="  ")
        self.tb_rp_svm.heading("#0", anchor="center")
        self.tb_rp_svm.column("#0", width="180")
        self.tb_rp_svm.column("#0", minwidth="10")
        self.tb_rp_svm.column("#0", stretch=False)
        self.tb_rp_svm.column("#0", anchor="w")

        self.frame_rp_random_forest = tk.LabelFrame(self.tab_report)
        self.frame_rp_random_forest.place(relx=0.349, rely=0.0, relheight=0.508
                                          , relwidth=0.332)
        self.frame_rp_random_forest.configure(relief='groove')
        self.frame_rp_random_forest.configure(text='''RandomForest''')

        self.tb_rp_randomforest = ScrolledTreeView(self.frame_rp_random_forest)
        self.tb_rp_randomforest.place(relx=0.0, rely=0.057, relheight=0.951
                                      , relwidth=1.157, bordermode='ignore')
        self.tb_rp_randomforest.configure(columns="Col1")
        # build_treeview_support starting.
        self.tb_rp_randomforest.heading("#0", text="  ")
        self.tb_rp_randomforest.heading("#0", anchor="center")
        self.tb_rp_randomforest.column("#0", width=180)
        self.tb_rp_randomforest.column("#0", minwidth=10)
        self.tb_rp_randomforest.column("#0", stretch=False)
        self.tb_rp_randomforest.column("#0", anchor="w")

        self.frame_rp_mlp = tk.LabelFrame(self.tab_report)
        self.frame_rp_mlp.place(relx=0.688, rely=0.0, relheight=0.508
                                , relwidth=0.312)
        self.frame_rp_mlp.configure(relief='groove')
        self.frame_rp_mlp.configure(text='''MLP''')
        self.tb_rp_mlp = ScrolledTreeView(self.frame_rp_mlp)
        self.tb_rp_mlp.place(relx=0.0, rely=0.057, relheight=0.951
                             , relwidth=1.228, bordermode='ignore')
        self.tb_rp_mlp.configure(columns="Col1")
        # build_treeview_support starting.
        self.tb_rp_mlp.heading("#0", text=" ")
        self.tb_rp_mlp.heading("#0", anchor="center")
        self.tb_rp_mlp.column("#0", width="180")
        self.tb_rp_mlp.column("#0", minwidth="10")
        self.tb_rp_mlp.column("#0", stretch=False)
        self.tb_rp_mlp.column("#0", anchor="w")

        self.frame_rp_knn = tk.LabelFrame(self.tab_report)
        self.frame_rp_knn.place(relx=0.0, rely=0.516, relheight=0.461
                                , relwidth=0.339)
        self.frame_rp_knn.configure(relief='groove')
        self.frame_rp_knn.configure(text='''KNN''')

        self.tb_rp_knn = ScrolledTreeView(self.frame_rp_knn)
        self.tb_rp_knn.place(relx=0.0, rely=0.064, relheight=0.918
                             , relwidth=1.132, bordermode='ignore')
        self.tb_rp_knn.configure(columns="Col1")
        # build_treeview_support starting.
        self.tb_rp_knn.heading("#0", text="  ")
        self.tb_rp_knn.heading("#0", anchor="center")
        self.tb_rp_knn.column("#0", width=180)
        self.tb_rp_knn.column("#0", minwidth=10)
        self.tb_rp_knn.column("#0", stretch=False)
        self.tb_rp_knn.column("#0", anchor="w")

        self.frame_rp_decision_tree = tk.LabelFrame(self.tab_report)
        self.frame_rp_decision_tree.place(relx=0.349, rely=0.516, relheight=0.461
                                          , relwidth=0.332)
        self.frame_rp_decision_tree.configure(relief='groove')
        self.frame_rp_decision_tree.configure(text='''DecisionTree''')
        self.tb_rp_decision_tree = ScrolledTreeView(self.frame_rp_decision_tree)
        self.tb_rp_decision_tree.place(relx=0.0, rely=0.061, relheight=0.918
                                       , relwidth=1.154, bordermode='ignore')
        self.tb_rp_decision_tree.configure(columns="Col1")
        # build_treeview_support starting.
        self.tb_rp_decision_tree.heading("#0", text="  ")
        self.tb_rp_decision_tree.heading("#0", anchor="center")
        self.tb_rp_decision_tree.column("#0", width=180)
        self.tb_rp_decision_tree.column("#0", minwidth=10)
        self.tb_rp_decision_tree.column("#0", stretch=False)
        self.tb_rp_decision_tree.column("#0", anchor="w")

        self.frame_rp_logistics = tk.LabelFrame(self.tab_report)
        self.frame_rp_logistics.place(relx=0.688, rely=0.516, relheight=0.461
                                      , relwidth=0.312)
        self.frame_rp_logistics.configure(relief='groove')
        self.frame_rp_logistics.configure(text='''Logistics''')
        self.tb_rp_logistics = ScrolledTreeView(self.frame_rp_logistics)
        self.tb_rp_logistics.place(relx=0.0, rely=0.061, relheight=0.918
                                   , relwidth=1.228, bordermode='ignore')
        self.tb_rp_logistics.configure(columns="Col1")
        # build_treeview_support starting.
        self.tb_rp_logistics.heading("#0", text=" ")
        self.tb_rp_logistics.heading("#0", anchor="center")
        self.tb_rp_logistics.column("#0", width=180)
        self.tb_rp_logistics.column("#0", minwidth=10)
        self.tb_rp_logistics.column("#0", stretch=False)
        self.tb_rp_logistics.column("#0", anchor="w")

        self.frame_cf_SVM_1 = tk.LabelFrame(self.tab_confussion_matrix)
        self.frame_cf_SVM_1.place(relx=0.0, rely=0.0, relheight=0.508
                                  , relwidth=0.341)
        self.frame_cf_SVM_1.configure(relief='groove')
        self.frame_cf_SVM_1.configure(text='''SVM''')
        self.cf_SVM = tk.Canvas(self.frame_cf_SVM_1)
        self.cf_SVM.place(relx=0.0, rely=0.061, relheight=0.927, relwidth=0.997
                          , bordermode='ignore')
        self.cf_SVM.configure(borderwidth="2")
        self.cf_SVM.configure(relief="ridge")
        self.cf_SVM.configure(selectbackground="#c4c4c4")
        self.frame_cf_random_forest = tk.LabelFrame(self.tab_confussion_matrix)
        self.frame_cf_random_forest.place(relx=0.339, rely=0.0, relheight=0.507
                                          , relwidth=0.342)
        self.frame_cf_random_forest.configure(relief='groove')
        self.frame_cf_random_forest.configure(text='''Randomforest''')
        self.cf_random_forest = tk.Canvas(self.frame_cf_random_forest)
        self.cf_random_forest.place(relx=0.0, rely=0.058, relheight=0.93
                                    , relwidth=0.995, bordermode='ignore')
        self.cf_random_forest.configure(borderwidth="2")
        self.cf_random_forest.configure(relief="ridge")
        self.cf_random_forest.configure(selectbackground="#c4c4c4")
        self.frame_cf_mlp = tk.LabelFrame(self.tab_confussion_matrix)
        self.frame_cf_mlp.place(relx=0.679, rely=0.0, relheight=0.508
                                , relwidth=0.34)
        self.frame_cf_mlp.configure(relief='groove')
        self.frame_cf_mlp.configure(text='''MLP''')
        self.cf_mlp = tk.Canvas(self.frame_cf_mlp)
        self.cf_mlp.place(relx=0.0, rely=0.061, relheight=0.927, relwidth=1.005
                          , bordermode='ignore')
        self.cf_mlp.configure(borderwidth="2")
        self.cf_mlp.configure(relief="ridge")
        self.cf_mlp.configure(selectbackground="#c4c4c4")
        self.frame_cf_knn = tk.LabelFrame(self.tab_confussion_matrix)
        self.frame_cf_knn.place(relx=0.0, rely=0.501, relheight=0.507
                                , relwidth=0.341)
        self.frame_cf_knn.configure(relief='groove')
        self.frame_cf_knn.configure(text='''KNN''')
        self.cf_knn = tk.Canvas(self.frame_cf_knn)
        self.cf_knn.place(relx=0.0, rely=0.058, relheight=0.927, relwidth=0.997
                          , bordermode='ignore')
        self.cf_knn.configure(borderwidth="2")
        self.cf_knn.configure(relief="ridge")
        self.cf_knn.configure(selectbackground="#c4c4c4")
        self.frame_cf_decision_tree = tk.LabelFrame(self.tab_confussion_matrix)
        self.frame_cf_decision_tree.place(relx=0.338, rely=0.507, relheight=0.507
                                          , relwidth=0.342)
        self.frame_cf_decision_tree.configure(relief='groove')
        self.frame_cf_decision_tree.configure(text='''DecisionTree''')
        self.cf_decision_tree = tk.Canvas(self.frame_cf_decision_tree)
        self.cf_decision_tree.place(relx=0.0, rely=0.058, relheight=0.927
                                    , relwidth=0.997, bordermode='ignore')
        self.cf_decision_tree.configure(borderwidth="2")
        self.cf_decision_tree.configure(relief="ridge")
        self.cf_decision_tree.configure(selectbackground="#c4c4c4")
        self.frame_cf_logistic = tk.LabelFrame(self.tab_confussion_matrix)
        self.frame_cf_logistic.place(relx=0.676, rely=0.507, relheight=0.507
                                     , relwidth=0.342)
        self.frame_cf_logistic.configure(relief='groove')
        self.frame_cf_logistic.configure(text='''Logistics''')
        self.cf_logistics = tk.Canvas(self.frame_cf_logistic)
        self.cf_logistics.place(relx=0.0, rely=0.058, relheight=0.927
                                , relwidth=0.997, bordermode='ignore')
        self.cf_logistics.configure(borderwidth="2")
        self.cf_logistics.configure(relief="ridge")
        self.cf_logistics.configure(selectbackground="#c4c4c4")
        self.op_features = tk.LabelFrame(self.tab_optional)
        self.op_features.place(relx=0.005, rely=0.012, relheight=0.114
                               , relwidth=0.989)
        self.op_features.configure(relief='groove')
        self.op_features.configure(text='''Feature Interest''')
        self.op_orX = tk.Checkbutton(self.op_features)
        self.op_orX.place(relx=0.01, rely=0.397, relheight=0.308, relwidth=0.064
                          , bordermode='ignore')
        self.op_orX.configure(activebackground="beige")
        self.op_orX.configure(anchor='w')
        self.op_orX.configure(compound='left')
        self.op_orX.configure(justify='left')
        self.op_orX.configure(selectcolor="#d9d9d9")
        self.op_orX.configure(text='''orX''')
        self.op_orX.configure(variable=self.checked_op_orX)
        self.op_orY = tk.Checkbutton(self.op_features)
        self.op_orY.place(relx=0.067, rely=0.397, relheight=0.308, relwidth=0.064
                          , bordermode='ignore')
        self.op_orY.configure(activebackground="beige")
        self.op_orY.configure(anchor='w')
        self.op_orY.configure(compound='left')
        self.op_orY.configure(justify='left')
        self.op_orY.configure(selectcolor="#d9d9d9")
        self.op_orY.configure(text='''orY''')
        self.op_orY.configure(variable=self.checked_op_orY)
        self.op_orZ = tk.Checkbutton(self.op_features)
        self.op_orZ.place(relx=0.124, rely=0.397, relheight=0.308, relwidth=0.065
                          , bordermode='ignore')
        self.op_orZ.configure(activebackground="beige")
        self.op_orZ.configure(anchor='w')
        self.op_orZ.configure(compound='left')
        self.op_orZ.configure(justify='left')
        self.op_orZ.configure(selectcolor="#d9d9d9")
        self.op_orZ.configure(text='''orZ''')
        self.op_orZ.configure(variable=self.checked_op_orZ)
        self.op_rX = tk.Checkbutton(self.op_features)
        self.op_rX.place(relx=0.191, rely=0.397, relheight=0.308, relwidth=0.064
                         , bordermode='ignore')
        self.op_rX.configure(activebackground="beige")
        self.op_rX.configure(anchor='w')
        self.op_rX.configure(compound='left')
        self.op_rX.configure(justify='left')
        self.op_rX.configure(selectcolor="#d9d9d9")
        self.op_rX.configure(text='''rX''')
        self.op_rX.configure(variable=self.checked_op_rX)
        self.op_rY = tk.Checkbutton(self.op_features)
        self.op_rY.place(relx=0.239, rely=0.397, relheight=0.308, relwidth=0.064
                         , bordermode='ignore')
        self.op_rY.configure(activebackground="beige")
        self.op_rY.configure(anchor='w')
        self.op_rY.configure(compound='left')
        self.op_rY.configure(justify='left')
        self.op_rY.configure(selectcolor="#d9d9d9")
        self.op_rY.configure(text='''rY''')
        self.op_rY.configure(variable=self.checked_op_rY)
        self.op_rZ = tk.Checkbutton(self.op_features)
        self.op_rZ.place(relx=0.287, rely=0.397, relheight=0.308, relwidth=0.064
                         , bordermode='ignore')
        self.op_rZ.configure(activebackground="beige")
        self.op_rZ.configure(anchor='w')
        self.op_rZ.configure(compound='left')
        self.op_rZ.configure(justify='left')
        self.op_rZ.configure(selectcolor="#d9d9d9")
        self.op_rZ.configure(text='''rZ''')
        self.op_rZ.configure(variable=self.checked_op_rZ)
        self.op_accX = tk.Checkbutton(self.op_features)
        self.op_accX.place(relx=0.344, rely=0.397, relheight=0.308
                           , relwidth=0.064, bordermode='ignore')
        self.op_accX.configure(activebackground="beige")
        self.op_accX.configure(anchor='w')
        self.op_accX.configure(compound='left')
        self.op_accX.configure(justify='left')
        self.op_accX.configure(selectcolor="#d9d9d9")
        self.op_accX.configure(text='''accX''')
        self.op_accX.configure(variable=self.checked_op_accX)
        self.op_accY = tk.Checkbutton(self.op_features)
        self.op_accY.place(relx=0.401, rely=0.397, relheight=0.308
                           , relwidth=0.065, bordermode='ignore')
        self.op_accY.configure(activebackground="beige")
        self.op_accY.configure(anchor='w')
        self.op_accY.configure(compound='left')
        self.op_accY.configure(justify='left')
        self.op_accY.configure(selectcolor="#d9d9d9")
        self.op_accY.configure(text='''accY''')
        self.op_accY.configure(variable=self.checked_op_accY)
        self.op_accZ = tk.Checkbutton(self.op_features)
        self.op_accZ.place(relx=0.458, rely=0.397, relheight=0.308
                           , relwidth=0.064, bordermode='ignore')
        self.op_accZ.configure(activebackground="beige")
        self.op_accZ.configure(anchor='w')
        self.op_accZ.configure(compound='left')
        self.op_accZ.configure(justify='left')
        self.op_accZ.configure(selectcolor="#d9d9d9")
        self.op_accZ.configure(text='''accZ''')
        self.op_accZ.configure(variable=self.checked_op_accZ)
        self.op_mX = tk.Checkbutton(self.op_features)
        self.op_mX.place(relx=0.705, rely=0.397, relheight=0.308, relwidth=0.065
                         , bordermode='ignore')
        self.op_mX.configure(activebackground="beige")
        self.op_mX.configure(anchor='w')
        self.op_mX.configure(compound='left')
        self.op_mX.configure(justify='left')
        self.op_mX.configure(selectcolor="#d9d9d9")
        self.op_mX.configure(text='''mX''')
        self.op_mX.configure(variable=self.checked_op_mX)
        self.op_mY = tk.Checkbutton(self.op_features)
        self.op_mY.place(relx=0.753, rely=0.397, relheight=0.308, relwidth=0.065
                         , bordermode='ignore')
        self.op_mY.configure(activebackground="beige")
        self.op_mY.configure(anchor='w')
        self.op_mY.configure(compound='left')
        self.op_mY.configure(justify='left')
        self.op_mY.configure(selectcolor="#d9d9d9")
        self.op_mY.configure(text='''mY''')
        self.op_mY.configure(variable=self.checked_op_mY)
        self.op_mZ = tk.Checkbutton(self.op_features)
        self.op_mZ.place(relx=0.801, rely=0.397, relheight=0.308, relwidth=0.065
                         , bordermode='ignore')
        self.op_mZ.configure(activebackground="beige")
        self.op_mZ.configure(anchor='w')
        self.op_mZ.configure(compound='left')
        self.op_mZ.configure(justify='left')
        self.op_mZ.configure(selectcolor="#d9d9d9")
        self.op_mZ.configure(text='''mZ''')
        self.op_mZ.configure(variable=self.checked_op_mZ)
        self.op_lux = tk.Checkbutton(self.op_features)
        self.op_lux.place(relx=0.848, rely=0.397, relheight=0.308, relwidth=0.065
                          , bordermode='ignore')
        self.op_lux.configure(activebackground="beige")
        self.op_lux.configure(anchor='w')
        self.op_lux.configure(compound='left')
        self.op_lux.configure(justify='left')
        self.op_lux.configure(selectcolor="#d9d9d9")
        self.op_lux.configure(text='''lux''')
        self.op_lux.configure(variable=self.checked_op_lux)
        self.op_gX = tk.Checkbutton(self.op_features)
        self.op_gX.place(relx=0.544, rely=0.397, relheight=0.308, relwidth=0.064
                         , bordermode='ignore')
        self.op_gX.configure(activebackground="beige")
        self.op_gX.configure(anchor='w')
        self.op_gX.configure(compound='left')
        self.op_gX.configure(justify='left')
        self.op_gX.configure(selectcolor="#d9d9d9")
        self.op_gX.configure(text='''gX''')
        self.op_gX.configure(variable=self.checked_op_gX)
        self.op_gY = tk.Checkbutton(self.op_features)
        self.op_gY.place(relx=0.601, rely=0.397, relheight=0.308, relwidth=0.064
                         , bordermode='ignore')
        self.op_gY.configure(activebackground="beige")
        self.op_gY.configure(anchor='w')
        self.op_gY.configure(compound='left')
        self.op_gY.configure(justify='left')
        self.op_gY.configure(selectcolor="#d9d9d9")
        self.op_gY.configure(text='''gY''')
        self.op_gY.configure(variable=self.checked_op_gY)
        self.op_gZ = tk.Checkbutton(self.op_features)
        self.op_gZ.place(relx=0.648, rely=0.397, relheight=0.308, relwidth=0.064
                         , bordermode='ignore')
        self.op_gZ.configure(activebackground="beige")
        self.op_gZ.configure(anchor='w')
        self.op_gZ.configure(compound='left')
        self.op_gZ.configure(justify='left')
        self.op_gZ.configure(selectcolor="#d9d9d9")
        self.op_gZ.configure(text='''gZ''')
        self.op_gZ.configure(variable=self.checked_op_gZ)
        self.op_soundlevel = tk.Checkbutton(self.op_features)
        self.op_soundlevel.place(relx=0.896, rely=0.397, relheight=0.308
                                 , relwidth=0.102, bordermode='ignore')
        self.op_soundlevel.configure(activebackground="beige")
        self.op_soundlevel.configure(anchor='w')
        self.op_soundlevel.configure(compound='left')
        self.op_soundlevel.configure(justify='left')
        self.op_soundlevel.configure(selectcolor="#d9d9d9")
        self.op_soundlevel.configure(text='''soundLevel''')
        self.op_soundlevel.configure(variable=self.checkeck_op_soundlevel)
        self.button_variance = tk.Button(self.tab_optional)
        self.button_variance.place(relx=0.146, rely=0.133, height=33, width=73)
        self.button_variance.configure(activebackground="beige")
        self.button_variance.configure(borderwidth="2")
        self.button_variance.configure(command=self.run_op_var)
        self.button_variance.configure(compound='left')
        self.button_variance.configure(text='''Variance''')
        self.button_mean = tk.Button(self.tab_optional)
        self.button_mean.place(relx=0.247, rely=0.133, height=33, width=73)
        self.button_mean.configure(activebackground="beige")
        self.button_mean.configure(borderwidth="2")
        self.button_mean.configure(command=self.run_op_mean)
        self.button_mean.configure(compound='left')
        self.button_mean.configure(text='''Mean''')
        self.button_std = tk.Button(self.tab_optional)
        self.button_std.place(relx=0.347, rely=0.133, height=33, width=133)
        self.button_std.configure(activebackground="beige")
        self.button_std.configure(borderwidth="2")
        self.button_std.configure(command=self.run_op_std)
        self.button_std.configure(compound='left')
        self.button_std.configure(text='''Standard Deviation''')
        self.button_rms = tk.Button(self.tab_optional)
        self.button_rms.place(relx=0.493, rely=0.133, height=33, width=73)
        self.button_rms.configure(activebackground="beige")
        self.button_rms.configure(borderwidth="2")
        self.button_rms.configure(command=self.run_op_rms)
        self.button_rms.configure(compound='left')
        self.button_rms.configure(text='''RMS''')
        self.button_zc = tk.Button(self.tab_optional)
        self.button_zc.place(relx=0.594, rely=0.133, height=33, width=73)
        self.button_zc.configure(activebackground="beige")
        self.button_zc.configure(borderwidth="2")
        self.button_zc.configure(command=self.run_op_zc)
        self.button_zc.configure(compound='left')
        self.button_zc.configure(text='''ZC''')
        self.button_sos = tk.Button(self.tab_optional)
        self.button_sos.place(relx=0.685, rely=0.133, height=33, width=73)
        self.button_sos.configure(activebackground="beige")
        self.button_sos.configure(borderwidth="2")
        self.button_sos.configure(command=self.run_op_sos)
        self.button_sos.configure(compound='left')
        self.button_sos.configure(text='''SOS''')
        self.button_cov = tk.Button(self.tab_optional)
        self.button_cov.place(relx=0.776, rely=0.134, height=33, width=93)
        self.button_cov.configure(activebackground="beige")
        self.button_cov.configure(borderwidth="2")
        self.button_cov.configure(command=self.run_op_cov)
        self.button_cov.configure(compound='left')
        self.button_cov.configure(text='''Covariance''')
        self.frame_op_output = tk.LabelFrame(self.tab_optional)
        self.frame_op_output.place(relx=0.009, rely=0.192, relheight=0.761
                                   , relwidth=0.986)
        self.frame_op_output.configure(relief='groove')
        self.frame_op_output.configure(text='''Output''')
        self.output_op = ScrolledText(self.frame_op_output)
        self.output_op.place(relx=0.0, rely=0.038, relheight=0.96, relwidth=1.002
                             , bordermode='ignore')
        self.output_op.configure(background="white")
        self.output_op.configure(font="TkTextFont")
        self.output_op.configure(insertborderwidth="3")
        self.output_op.configure(selectbackground="#c4c4c4")
        self.output_op.configure(wrap="none")
        self.checked_list_optional_tab = {}
        self.checked_list_model = []

    def run_describe(self):
        check_list_statistic_tab = {
            'orX': self.checked_orX,
            'orY': self.checked_orY,
            'orZ': self.checked_orZ,
            'rX': self.checked_rX,
            'rY': self.checked_rY,
            'rZ': self.checked_rZ,
            'accX': self.checked_accX,
            'accY': self.checked_accY,
            'accZ': self.checked_accZ,
            'gX': self.checked_gX,
            'gY': self.checked_gY,
            'gZ': self.checked_gZ,
            'mX': self.checked_mX,
            'mY': self.checked_mY,
            'mZ': self.checked_mZ,
            'lux': self.checked_lux,
            'soundLevel': self.checked_sound_level
        }

        checked_list_statistic_tab = [key for key, var in check_list_statistic_tab.items() if var.get() == 1]
        describe = self.user.info(checked_list_statistic_tab)
        self.frame_statistic_table.delete(*self.frame_statistic_table.get_children())
        for index, row in describe.iterrows():
            item_id = str(index)
            values = tuple(row[column] for column in describe.columns)
            self.frame_statistic_table.insert("", "end", text=item_id, values=values)

    def run_clean_data(self):
        self.label_clean_completed.configure(text='''Waitting.....''')
        self.user.clean()
        self.label_clean_completed.configure(text='Clean completed!!!!')

    def run_train(self):
        self.model_status.configure(text='''Waiting...''')
        self.model_status.update()
        self.frame_text_best_params.configure(state="normal")
        self.frame_text_best_params.delete("1.0", "end")
        self.frame_text_best_params.update()

        check_list_model = {
            'SVM': self.user.model_svm,
            'Random Forest': self.user.model_random_forest,
            'MLP': self.user.model_multi_layer_perception_neural_network,
            'KNN': self.user.model_knn,
            'Logistic Classifier': self.user.model_logistic,
            'Decision Tree': self.user.model_decision_tree
        }
        model_vars = {
            'SVM': self.checked_SVM,
            'Random Forest': self.checked_random_forest,
            'MLP': self.checked_mlp,
            'KNN': self.checked_knn,
            'Logistic Classifier': self.checked_logistic,
            'Decision Tree': self.checked_decision_tree
        }
        self.checked_list_model = [check_list_model[key] for key, var in model_vars.items() if var.get() == 1]

        best_params_string = ''
        for run_models in self.checked_list_model:
            best_params_string += run_models()
        self.frame_text_best_params.insert("1.0", best_params_string)

        self.model_status.configure(text='''Train completed!!!''')
        self.frame_text_best_params.update()
        self.frame_text_best_params.configure(state="disabled")
        self.model_status.update()

        # self.user.pre_unbalanced_classification()

    def run_predict(self):
        self.model_status.configure(text="")
        self.model_status.update()
        self.frame_text_best_params.configure(state="normal")
        self.frame_text_best_params.delete("1.0", "end")

        best_params_string = ''
        dict_model = {
            'SVM': 'model_svm.pkl',
            'Random Forest': 'model_randomforest.pkl',
            'MLP': 'model_mlp.pkl',
            'KNN': 'model_knn.pkl',
            'Logistics': 'model_logistic.pkl',
            'Decision Tree': 'model_decisiontree.pkl'

        }
        model_vars = {
            'SVM': self.checked_SVM,
            'Random Forest': self.checked_random_forest,
            'MLP': self.checked_mlp,
            'KNN': self.checked_knn,
            'Logistics': self.checked_logistic,
            'Decision Tree': self.checked_decision_tree

        }
        self.checked_list_model = [key for key, var in model_vars.items() if var.get() == 1]

        for selected_model in self.checked_list_model:
            with open(dict_model[selected_model], 'rb') as file:
                loaded_model = pickle.load(file)
            if hasattr(loaded_model, 'best_params_'):
                best_params = loaded_model.best_params_

                best_params_string += f"Best parameters for {selected_model}: \n{best_params}\n"
            # Use the loaded model for predictions
            self.y_predicts[selected_model] = loaded_model.predict(self.user.X_test)
        self.frame_text_best_params.insert(tk.END, best_params_string)
        self.model_status.configure(text='''Predict Complete!!!''')
        self.model_status.update()
        self.frame_text_best_params.update_idletasks()
        self.frame_text_best_params.configure(state="disabled")

    def run_report(self):
        frames = {
            'SVM': self.tb_rp_svm,
            'Random Forest': self.tb_rp_randomforest,
            'MLP': self.tb_rp_mlp,
            'KNN': self.tb_rp_knn,
            'Logistics': self.tb_rp_logistics,
            'Decision Tree': self.tb_rp_decision_tree
        }
        for name in frames:
            frames[name].delete(*frames[name].get_children())
        for name in self.checked_list_model:
            report = self.user.report(self.user.y_test, self.y_predicts[name])
            frames[name].configure(columns=list(report.columns))

            for column in report.columns:
                frames[name].heading(column, text=column)
                frames[name].column(column, width=70)
                frames[name].column(column, minwidth=10)
                frames[name].column(column, stretch=False)
                frames[name].column(column, anchor='w')
            for index, row in report.iterrows():
                item_id = str(index)
                values = tuple(row[column] for column in report.columns)
                frames[name].insert("", "end", text=item_id, values=values)
        pass

    def run_confusion_matrix(self):
        dict_model = {
            'SVM': 'model_svm.pkl',
            'Random Forest': 'model_randomforest.pkl',
            'MLP': 'model_mlp.pkl',
            'KNN': 'model_knn.pkl',
            'Logistics': 'model_logistic.pkl',
            'Decision Tree': 'model_decisiontree.pkl'
        }

        frames = {
            'SVM': self.cf_SVM,
            'Random Forest': self.cf_random_forest,
            'MLP': self.cf_mlp,
            'KNN': self.cf_knn,
            'Logistics': self.cf_logistics,
            'Decision Tree': self.cf_decision_tree
        }

        for name in self.checked_list_model:
            with open(dict_model[name], 'rb') as file:
                loaded_model = pickle.load(file)
            fig = self.user.evaluation(name, loaded_model, self.y_predicts[name])

            # Create the canvas for the current model
            canvas = tk.Canvas(frames[name])
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Create the FigureCanvasTkAgg object and associate it with the canvas
            canvas1 = FigureCanvasTkAgg(fig, master=canvas)
            canvas1.draw()
            canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.frame_cf_SVM_1.mainloop()
        self.frame_cf_random_forest.mainloop()
        self.frame_cf_decision_tree.mainloop()
        self.frame_cf_knn.mainloop()
        self.frame_cf_decision_tree.mainloop()
        self.frame_cf_logistic.mainloop()

    def run_frequency_dependence(self):

        ax1 = self.user.frequency()
        canvas1 = FigureCanvasTkAgg(ax1.figure, master=self.frame_frequency)
        canvas1.draw()
        canvas1.get_tk_widget().pack()

        fig2 = self.user.dependencies()
        canvas2 = FigureCanvasTkAgg(fig2, master=self.frame_dependances)
        canvas2.draw()
        canvas2.get_tk_widget().pack()

        self.tab_freq_depend.mainloop()

    def optional_checked(self):
        self.check_list_optional_tab = {
            'orX': self.checked_op_orX,
            'orY': self.checked_op_orY,
            'orZ': self.checked_op_orZ,
            'rX': self.checked_op_rX,
            'rY': self.checked_op_rY,
            'rZ': self.checked_op_rZ,
            'accX': self.checked_op_accX,
            'accY': self.checked_op_accY,
            'accZ': self.checked_op_accZ,
            'gX': self.checked_op_gX,
            'gY': self.checked_op_gY,
            'gZ': self.checked_op_gZ,
            'mX': self.checked_op_mX,
            'mY': self.checked_op_mY,
            'mZ': self.checked_op_mZ,
            'lux': self.checked_op_lux,
            'soundLevel': self.checkeck_op_soundlevel
        }
        self.checked_list_optional_tab = [key for key, var in self.check_list_optional_tab.items() if var.get() == 1]

    def run_op_var(self):
        self.output_op.configure(state='normal')
        self.output_op.delete("1.0", "end")
        self.optional_checked()
        self.output_op.insert("1.0", f'{self.user.var(self.checked_list_optional_tab)}')
        self.output_op.update()
        self.output_op.configure(state='disable')

    def run_op_mean(self):
        self.output_op.configure(state='normal')
        self.output_op.delete("1.0", "end")
        self.optional_checked()
        self.output_op.insert("1.0", f'{self.user.mean(self.checked_list_optional_tab)}')
        self.output_op.update()
        self.output_op.configure(state='disable')

    def run_op_std(self):
        self.output_op.configure(state='normal')
        self.output_op.delete("1.0", "end")
        self.optional_checked()
        self.output_op.insert("1.0", f'{self.user.std(self.checked_list_optional_tab)}')
        self.output_op.update()
        self.output_op.configure(state='disable')

    def run_op_rms(self):
        self.output_op.configure(state='normal')
        self.output_op.delete("1.0", "end")
        self.optional_checked()
        self.output_op.insert("1.0", f'{self.user.rms(self.checked_list_optional_tab)}')
        self.output_op.update()
        self.output_op.configure(state='disable')

    def run_op_zc(self):
        self.output_op.configure(state='normal')
        self.output_op.delete("1.0", "end")
        self.optional_checked()
        self.output_op.insert("1.0", f'{self.user.zero_crossing(self.checked_list_optional_tab)}')
        self.output_op.update()
        self.output_op.configure(state='disable')

    def run_op_sos(self):
        self.output_op.configure(state='normal')
        self.output_op.delete("1.0", "end")
        self.optional_checked()
        self.output_op.insert("1.0", f'{self.user.sos()}')
        self.output_op.update()
        self.output_op.configure(state='disable')

    def run_op_cov(self):
        self.output_op.configure(state='normal')
        self.output_op.delete("1.0", "end")
        self.optional_checked()
        self.output_op.insert("1.0", f'{self.user.cov(self.checked_list_optional_tab)}')
        self.output_op.update()
        self.output_op.configure(state='disable')


class AutoScroll(object):
    '''Configure the scrollbars for a widget.'''

    def __init__(self, master):
        #  Rozen. Added the try-except clauses so that this class
        #  could be used for scrolled entry widget for which vertical
        #  scrolling is not supported. 5/7/14.
        try:
            vsb = ttk.Scrollbar(master, orient='vertical', command=self.yview)
        except:
            pass
        hsb = ttk.Scrollbar(master, orient='horizontal', command=self.xview)
        try:
            self.configure(yscrollcommand=self._autoscroll(vsb))
        except:
            pass
        self.configure(xscrollcommand=self._autoscroll(hsb))
        self.grid(column=0, row=0, sticky='nsew')
        try:
            vsb.grid(column=1, row=0, sticky='ns')
        except:
            pass
        hsb.grid(column=0, row=1, sticky='ew')
        master.grid_columnconfigure(0, weight=1)
        master.grid_rowconfigure(0, weight=1)
        # Copy geometry methods of master  (taken from ScrolledText.py)
        methods = tk.Pack.__dict__.keys() | tk.Grid.__dict__.keys() \
                  | tk.Place.__dict__.keys()
        for meth in methods:
            if meth[0] != '_' and meth not in ('config', 'configure'):
                setattr(self, meth, getattr(master, meth))

    @staticmethod
    def _autoscroll(sbar):
        '''Hide and show scrollbar as needed.'''

        def wrapped(first, last):
            first, last = float(first), float(last)
            if first <= 0 and last >= 1:
                sbar.grid_remove()
            else:
                sbar.grid()
            sbar.set(first, last)

        return wrapped

    def __str__(self):
        return str(self.master)


def _create_container(func):
    '''Creates a ttk Frame with a given master, and use this new frame to
    place the scrollbars and the widget.'''

    def wrapped(cls, master, **kw):
        container = ttk.Frame(master)
        container.bind('<Enter>', lambda e: _bound_to_mousewheel(e, container))
        container.bind('<Leave>', lambda e: _unbound_to_mousewheel(e, container))
        return func(cls, container, **kw)

    return wrapped


class ScrolledText(AutoScroll, tk.Text):
    '''A standard Tkinter Text widget with scrollbars that will
    automatically show/hide as needed.'''

    @_create_container
    def __init__(self, master, **kw):
        tk.Text.__init__(self, master, **kw)
        AutoScroll.__init__(self, master)


class ScrolledTreeView(AutoScroll, ttk.Treeview):
    '''A standard ttk Treeview widget with scrollbars that will
    automatically show/hide as needed.'''

    @_create_container
    def __init__(self, master, **kw):
        ttk.Treeview.__init__(self, master, **kw)
        AutoScroll.__init__(self, master)


import platform


def _bound_to_mousewheel(event, widget):
    child = widget.winfo_children()[0]
    if platform.system() == 'Windows' or platform.system() == 'Darwin':
        child.bind_all('<MouseWheel>', lambda e: _on_mousewheel(e, child))
        child.bind_all('<Shift-MouseWheel>', lambda e: _on_shiftmouse(e, child))
    else:
        child.bind_all('<Button-4>', lambda e: _on_mousewheel(e, child))
        child.bind_all('<Button-5>', lambda e: _on_mousewheel(e, child))
        child.bind_all('<Shift-Button-4>', lambda e: _on_shiftmouse(e, child))
        child.bind_all('<Shift-Button-5>', lambda e: _on_shiftmouse(e, child))


def _unbound_to_mousewheel(event, widget):
    if platform.system() == 'Windows' or platform.system() == 'Darwin':
        widget.unbind_all('<MouseWheel>')
        widget.unbind_all('<Shift-MouseWheel>')
    else:
        widget.unbind_all('<Button-4>')
        widget.unbind_all('<Button-5>')
        widget.unbind_all('<Shift-Button-4>')
        widget.unbind_all('<Shift-Button-5>')


def _on_mousewheel(event, widget):
    if platform.system() == 'Windows':
        widget.yview_scroll(-1 * int(event.delta / 120), 'units')
    elif platform.system() == 'Darwin':
        widget.yview_scroll(-1 * int(event.delta), 'units')
    else:
        if event.num == 4:
            widget.yview_scroll(-1, 'units')
        elif event.num == 5:
            widget.yview_scroll(1, 'units')


def _on_shiftmouse(event, widget):
    if platform.system() == 'Windows':
        widget.xview_scroll(-1 * int(event.delta / 120), 'units')
    elif platform.system() == 'Darwin':
        widget.xview_scroll(-1 * int(event.delta), 'units')
    else:
        if event.num == 4:
            widget.xview_scroll(-1, 'units')
        elif event.num == 5:
            widget.xview_scroll(1, 'units')


def start_up():
    python_gui_as2_support.main()


if __name__ == '__main__':
    python_gui_as2_support.main()
