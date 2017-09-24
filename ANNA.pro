TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    activation_functions.cpp \
    ANN.cpp \
    Layer.cpp \
    Neuron.cpp

HEADERS += \
    activation_functions.h \
    ANN.h \
    Layer.h \
    Neuron.h
