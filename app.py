from flask import Flask, render_template, url_for, redirect, request
import pandas as pd
app = Flask(__name__)

@app.route('/')
def index():
    df = pd.read_csv('output/user_pc_ct.csv')
    legend = 'USER vs Async Score'
    legend1 = "User vs Threat Confirmation"
    labels1 = list(df.user)
    values1 = list(df.ascore)
    values2 = list(df.threat)

    df = pd.read_csv('output/device_file_full_result.csv')
    legend = 'USER vs Async Score'
    legend1 = "User vs Threat Confirmation"
    labels2 = list(df.user)
    values3 = list(df.ascore)
    values4 = list(df.threat)

    df = pd.read_csv('output/psychometric_result.csv')
    legend = 'USER vs Async Score'
    legend1 = "User vs Threat Confirmation"
    labels3 = list(df.user)
    values5 = list(df.ascore)
    values6 = list(df.threat)

    df = pd.read_csv('output/user_log_result.csv')
    legend = 'USER vs Async Score'
    legend1 = "User vs Threat Confirmation"
    labels4 = list(df.user)
    values7 = list(df.ascore)
    values8 = list(df.threat)

    df = pd.read_csv('output/all_parameters_result.csv')
    legend = 'USER vs Async Score'
    legend1 = "User vs Threat Confirmation"
    labels5 = list(df.user)
    values9 = list(df.ascore)
    values10 = list(df.threat)

    
    return render_template('index.html', values1=values1, labels1=labels1, legend=legend,values2=values2, legend1=legend1, labels2=labels2, values3=values3,values4=values4, labels3=labels3, values5=values5,values6=values6, labels4=labels4, values7=values7,values8=values8, labels5=labels5, values9=values9,values10=values10)
if __name__ == "__main__":
    app.run(debug=True)
