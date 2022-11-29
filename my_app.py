from flask import Flask, url_for, request, redirect, render_template
from werkzeug.utils import secure_filename
from Model import ApplanationSegmentation
import matplotlib
import numpy as np
from main import preprocess
import cv2
import torch
import os


# определили экземпляр класса flask
app = Flask(__name__)

# ограничение объема файла в байтах
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
UPLOAD_FOLDER = "F:/Project_py/seg_app/static/images"
RELATIVE_FILE_PATH = "static/images"
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# главная страница
@app.route("/index")
@app.route('/')
def main_page():
    # возвращает шаблон из папки templates
    return render_template("base.html")


# загрузка изображения и отображение
@app.route('/upload', methods=["POST", "GET"])
def upload():
    if request.method == "POST":
        # это поле ассоциировано загруженным на сервер изображением
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('show',
                                    filename=filename))
    return ''


@app.route('/<filename>')
def show(filename):
    path = RELATIVE_FILE_PATH + '/' + str(filename)
    return render_template('show.html', name=path)


# сделать универсальнее, чтобы люьое из изображений можно было подгружать
# как пробромсить filename?
@app.route('/segmentation', methods=["POST", "GET"])
def segmentation():

    filename = RELATIVE_FILE_PATH + '/' + "0.bmp"
    data = cv2.imread(filename)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    standardized_scan = preprocess.standardize(preprocess.normalize(data))

    # # загрузка модели
    model = ApplanationSegmentation.load_from_checkpoint("epoch=4-step=2999.ckpt")

    # Got 3D input, but bilinear mode needs 4D input
    data = torch.tensor(standardized_scan).to(device).unsqueeze(0).unsqueeze(0).float()
    #
    with torch.no_grad():
        pred = model(data)
    pred = pred.cpu().numpy()

    # приводим к 800x800
    img_test = np.array(pred)
    img_test = np.squeeze(img_test)
    path = os.path.join(app.config['UPLOAD_FOLDER'], "segment_" + "0.bmp")
    matplotlib.image.imsave(path, img_test)
    return render_template('segmentation.html', name=path)


# запуск flask приложения
if __name__ == '__main__':
    # app.run(host, port, debug, options)
    # запуск локального веб-сервера 127.0.0.1:5000
    # доступ только с локальной машины
    # host='0.0.0.0' - позволит принимать запросы с интерфейса, поделюченного к обшедоступной сети
    # port=8080
    app.run(debug=True)