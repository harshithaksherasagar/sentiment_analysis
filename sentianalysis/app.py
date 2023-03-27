from colorama import Fore
from termcolor import colored
from transformers import pipeline
from flask import Flask, request, redirect, url_for, flash, render_template
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
import urllib.request
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#import seaborn as sns

app = Flask(__name__)
f= colored('blue')

@app.route('/submit', methods=['POST'])
def my_form_post():
    if request.method == 'POST':
       classifier = pipeline('sentiment-analysis')
       text1 = request.form.get('etext')
       f=classifier(text1)
       v=list(f[0].values())
       df=pd.DataFrame(v)
       if df[0][0]=="POSITIVE":
           x="NEGATIVE"
       else:
           x="POSITIVE"
       dt=pd.DataFrame([df[0][1],1-(df[0][1])],index=[df[0][0],x])
       dt = dt.plot(kind='pie',subplots=True,figsize=(8,8))
       fig=dt[0].figure
       img = io.BytesIO()
       fig.savefig(img,format='png')
       img.seek(0)
       plot_data = urllib.parse.quote(base64.b64encode(img.getvalue()).decode('utf-8'))
       return render_template('form.html',output=f,plot_url=plot_data)


@app.route('/')
def my_form():
    return render_template('form.html')



if __name__ == "__main__":
    app.run(debug=True)
    # if df[0][0]=="POSITIVE":
    #        x="NEGATIVE"
    # else:
    #    x="POSITIVE"
    #    dt=pd.DataFrame([df[0][1],1-(df[0][1])],index=[df[0][0],x])
    #    dt=dt.plot(kind='pie',sed=True)