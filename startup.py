import numpy as np
from scipy.signal import fftconvolve
from scipy.fft import fft, fftshift
from flask import Flask, render_template, request, redirect, url_for, jsonify

def ft(x):
    sp = fftshift(fft(x))
    return sp


def sincinterp(x):
    N = len(x)
    y = np.zeros(2 * N - 1, dtype=x.dtype)
    y[:2 * N:2] = x
    xint = fftconvolve( y[:2 * N], np.sinc(np.arange(-(2 * N - 3), (2 * N - 2)).T / 2),)
    return xint[2 * N - 3: -2 * N + 3]

def frft(f, a):
        ret = np.zeros_like(f, dtype=np.complex)
        f = f.copy().astype(np.complex)
        N = len(f)
        shft = np.fmod(np.arange(N) + np.fix(N / 2), N).astype(int)
        sN = np.sqrt(N)
        a = np.remainder(a, 4.0)

        # Special cases
        if a == 0.0:
            return f
        if a == 1.0:
            ret[shft] = np.fft.fft(f[shft]) / sN
            return ret
        if a < 0.5:
            a = a + 1
            f[shft] = np.fft.ifft(f[shft]) * sN

        # the general case for 0.5 < a < 1.5
        alpha = a * np.pi / 2
        tana2 = np.tan(alpha / 2)
        sina = np.sin(alpha)
        f = np.hstack((np.zeros(N - 1), sincinterp(f), np.zeros(N - 1))).T

        # chirp premultiplication
        chrp = np.exp(-1j * np.pi / N * tana2 / 4 * np.arange(-2 * N + 2, 2 * N - 1).T ** 2)
        f = chrp * f

        # chirp convolution
        c = np.pi / N / sina / 4
        ret = fftconvolve(np.exp(1j * c * np.arange(-(4 * N - 4), 4 * N - 3).T ** 2), f)
        ret = ret[4 * N - 4:8 * N - 7] * np.sqrt(c / np.pi)

        # chirp post multiplication
        ret = chrp * ret

        # normalizing constant
        ret = np.exp(-1j * (1 - a) * np.pi / 4) * ret[N - 1:-N + 1:2]
        return ret



def triangle(length, amplitude):
     section = length // 4
     for direction in (1, -1):
         for i in range(section):
             yield i * (amplitude / section) * direction
         for i in range(section):
             yield (amplitude - (i * (amplitude / section))) * direction


def rectangle(x1, x2, length):
    x = np.linspace(0, length, length, dtype=np.float)
    mask = (x>=x1) & (x <=x2)
    y = np.where(mask, 1, 0)
    y = [float(i) for i in y]
    return y


def sine(cycles, length):
    Fs = length
    f = cycles
    sample = length
    x = np.arange(sample)
    y = np.sin(2 * np.pi * f * x / Fs)
    return y



app = Flask(__name__)

@app.route('/')
def index():
    signal = rectangle(45, 80, 120)
    signal = np.array(signal)
    partialFt = frft(signal,0.3)
    fullFt = ft(signal)
    partialFtData = {"real": list(partialFt.real), "imag": list(partialFt.imag)}
    fullFtData = {"real": list(fullFt.real), "imag": list(fullFt.imag)}
    signalData = list(signal)
    labels = [i for i in range(len(signal))]
    return render_template('index.html', labels=labels, signalData=signalData, partialFtData=partialFtData, fullFtData=fullFtData)


@app.route('/update-range', methods=['POST'])
def ModifyRange():
    signalShape = request.form.get('signal')
    alpha = float(request.form['alpha'])
    signal = None
    if signalShape == "triangle":
        signal = list(triangle(240, 1))[:120]
    elif signalShape == "rectangle":
        signal = rectangle(45, 80, 120)
    elif signalShape == "sine":
        signal = list(sine(2, 125))
    signal = np.array(signal)
    partialFt = frft(signal, alpha)
    realData = list(partialFt.real)
    imagData = list(partialFt.imag)
    return {"realData":realData, "imagData":imagData}


@app.route('/update-shape', methods=['POST'])
def ModifyShape():
    signalShape = request.form.get('signal')
    alpha = float(request.form['alpha'])
    signal = None
    if signalShape == "triangle":
        signal = list(triangle(240, 1))[:120]
    elif signalShape == "rectangle":
        signal = rectangle(45, 80, 120)
    elif signalShape == "sine":
        signal = list(sine(2, 120))
    signal = np.array(signal)
    partialFt = frft(signal, alpha)
    fullFt = ft(signal)
    partialFtData = {"real": list(partialFt.real), "imag": list(partialFt.imag)}
    fullFtData = {"real": list(fullFt.real), "imag": list(fullFt.imag)}
    signalData = list(signal)
    return {"signalData": signalData, "partialFtData": partialFtData, "fullFtData": fullFtData}

if __name__ == '__main__':
    app.run(debug=True)