import pandas as pd
import io
import os
import base64
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import t
import matplotlib.pyplot as plt
import os
from flask import Flask, render_template, request, redirect, flash
import pandas as pd
import statistics
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Untuk penggunaan flash messages

# Tentukan folder untuk menyimpan file yang diupload
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tentukan ekstensi file yang diizinkan
ALLOWED_EXTENSIONS = {'xls', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/hasil', methods=['POST'])
def upload_file():
    if 'excelFile' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['excelFile']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            df = pd.read_excel(file_path)
            data = df.values.tolist()
            columns = df.columns.tolist()

            k = 4
            n = len(df)- k
            df["total_x1"] = df.iloc[:, 6:14].sum(axis=1)
            df["total_x2"] = df.iloc[:, 14:19].sum(axis=1)
            df["total_x3"] = df.iloc[:, 19:22].sum(axis=1)
            df["total_x4"] = df.iloc[:, 22:24].sum(axis=1)
            df["total_y"] = df.iloc[:, 24:26].sum(axis=1)
            Xvar = df[['total_x1','total_x2','total_x3','total_x4']]
            Xvar = sm.add_constant(Xvar)
            Yvar = df['total_y']
            olsmod = sm.OLS(Yvar,Xvar).fit()
            df['Ypredict'] = round(olsmod.predict(Xvar),4)
            df['Residual'] = round(olsmod.resid,4)

            def uji_normalitas_residual():
                n_stats, p_val = sm.stats.diagnostic.kstest_normal(
                df['Residual'],dist='norm', pvalmethod='table')
                print('P_value Kolmogorov-Smirnov :', round(p_val,3))

                if p_val > 0.05:
                    return ("Data terdistribusi normal")
                elif p_val < 0.05:
                    return ("Data terdistribusi tidak normal")
                
            def uji_multikol():
                vif_data = pd.DataFrame()
                vif_data["variabel"] = Xvar.columns
                vif_data["VIF"] = [variance_inflation_factor(Xvar.values, i) for i in range(len(Xvar.columns))]
                vif_data['Kesimpulan'] = ['Terjadi multikol' if vif > 10 else 'Tidak terjadi multikol' for vif in vif_data["VIF"]]
                vif_data = vif_data.drop(vif_data[vif_data['variabel'] == 'const'].index, errors='ignore')
                return vif_data.to_html(classes='table table-striped', index=False)    

            def uji_hetero():
                plt.figure()
                plot = sns.residplot(x=df["Ypredict"], y=df["Residual"])
                plot.set(title='Predict vs Residual')

                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plt.close()

                img_base64 = base64.b64encode(img.getvalue()).decode()
                return img_base64

            def auto_korelasi():
                durbinWatson = durbin_watson(df["Residual"])
                # print("Nilai Auto Korelasi :", durbinWatson)
                tb_dw = pd.read_excel('tabel_dw.xlsx')
                n = len(df)
                dl_du = tb_dw[tb_dw['n'] == n]
                dl_value = dl_du['dl'].values[0]
                du_value = dl_du['du'].values[0]
                # print(f"N = {n} maka DL = {dl_value} dan DU = {du_value}")

                if durbinWatson >= du_value and durbinWatson < 4 - du_value:
                    cetak = "Tidak terjadi Auto Korelasi"
                elif durbinWatson < dl_value and durbinWatson > 4 - dl_value:
                    cetak = "Terjadi auto korelasi"
                return cetak                
            
            def uji_t():
                df_t = pd.DataFrame(columns=['dk', 'α=0.05'])
                df_t['dk'] = [i for i in range(1, len(df))]
                df_t['α=0.05'] = [round(t.ppf(0.975, df), 3) for df in df_t['dk']]
                nilai_t = df_t[df_t['dk'] == n-1]['α=0.05'].values[0]

                total_x1 = olsmod.tvalues['total_x1']
                total_x2 = olsmod.tvalues['total_x2']
                total_x3 = olsmod.tvalues['total_x3']
                total_x4 = olsmod.tvalues['total_x4']
                const = olsmod.tvalues['const']

                hasil = []
                if total_x1 > nilai_t:
                    hasil.append("Nilai total_x1 berpengaruh pada Y.")
                else:
                    hasil.append("Nilai total_x1 tidak berpengaruh pada Y.")

                if total_x2 > nilai_t:
                    hasil.append("Nilai total_x2 berpengaruh pada Y.")
                else:
                    hasil.append("Nilai total_x2 tidak berpengaruh pada Y.")

                if total_x3 > nilai_t:
                    hasil.append("Nilai total_x3 berpengaruh pada Y.")
                else:
                    hasil.append("Nilai total_x3 tidak berpengaruh pada Y.")

                if total_x4 > nilai_t:
                    hasil.append("Nilai total_x4 berpengaruh pada Y.")
                else:
                    hasil.append("Nilai total_x4 tidak berpengaruh pada Y.")

                return hasil

            def uji_regresi():
                # Nilai Linear Regression
                bebas = ['total_x1', 'total_x2', 'total_x3', 'total_x4']
                X = df[bebas]
                y = df['total_y']

                model = LinearRegression().fit(X, y)

                koefisien = model.coef_
                intercept = model.intercept_

                hasil_regresi = {
                    "intercept": intercept,
                    "koefisien": list(zip(bebas, koefisien))
                }
                return hasil_regresi

            def penerimaan():
                nilai_max = 5 * len(df)
                df.loc["Jumlah"] = df.iloc[:, 6:26].sum()
                df.loc['%per item'] = (df.loc['Jumlah'] / nilai_max) * 100

                def kesimpulan(rata, minsd, plussd):
                    if rata < minsd:
                        return "Rendah"
                    elif rata < plussd:
                        return "Sedang"
                    else:
                        return "Tinggi"

                rata_persen_x1 = (df.iloc[96,6:14].sum()) / 8
                rata_persen_x2 = (df.iloc[96,14:19].sum()) / 5
                rata_persen_x3 = (df.iloc[96,19:22].sum()) / 3
                rata_persen_x4 = (df.iloc[96,22:24].sum()) / 2
                rata_persen_y = (df.iloc[96,24:26].sum()) / 2

                mean = df.loc["%per item"].sum() / 20
                stdevi = statistics.stdev(df.iloc[96,6:26].values)
                minsd = round(mean - stdevi)
                plussd = round(mean + stdevi)

                kesimpulan_akhir = round((rata_persen_x1 + rata_persen_x2 + rata_persen_x3 + rata_persen_x4 + rata_persen_y) / 5)
                label_class = "anto"

                def k_akhir(rata, minsd, plussd):
                    if rata < minsd:
                        label_class = "danger"
                        return f"Tidak Diterima, karena Nilai mean dari Rata % tiap item = {kesimpulan_akhir}%, dan Dikategorikan = {kesimpulan(kesimpulan_akhir, minsd, plussd)}"
                    elif rata < plussd:
                        label_class = "warning"
                        return f"Kurang Untuk Diterima, karena Nilai mean dari Rata % tiap item = {kesimpulan_akhir}%, dan Dikategorikan = {kesimpulan(kesimpulan_akhir, minsd, plussd)}"
                    else:
                        label_class = "success"
                        return f"Diterima, karena Nilai mean dari Rata % tiap item = {kesimpulan_akhir}%, dan Dikategorikan = {kesimpulan(kesimpulan_akhir, minsd, plussd)}"
                return k_akhir(kesimpulan_akhir, minsd, plussd)

            return render_template('display.html', data=data, columns=columns, normalitas = uji_normalitas_residual(), multikol = uji_multikol(), hetero = uji_hetero(),
                    korelasi = auto_korelasi(), t_result = uji_t(), regresi = uji_regresi(), penerimaan=penerimaan())
        except Exception as e:
            flash(f'An error occurred while processing the file: {e}')
            return
    else:
        flash('Allowed file types are .xls, .xlsx')
        return redirect(request.url)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
