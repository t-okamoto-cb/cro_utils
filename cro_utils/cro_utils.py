import pandas as pd
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import bigquery
from google.cloud import bigquery_storage
from IPython.display import display, HTML
import subprocess
import time
from pathlib import Path
import gspread
from google.auth import default as google_auth_default
from gspread_dataframe import set_with_dataframe
import datetime

# Evaluate regression
def evaluate_regression(t, y, yyplot=True, yyplot_png=True, yyplot_svg=False, show_scores=True, yyplot_filename="yyplot"):
    # Error check
    if len(t) != len(y):
        raise ValueError("len(t) != len(y) : " + str(len(t)) + " != " + str(len(y)))

    # Stats
    output = {}
    output["N"] = len(t)
    output["t_min"] = np.min(t)
    output["t_med"] = np.median(t)
    output["t_mean"] = np.mean(t)
    output["t_max"] = np.max(t)
    output["t_var"] = np.var(t)
    output["t_std"] = np.std(t)
    output["y_min"] = np.min(y)
    output["y_med"] = np.median(y)
    output["y_mean"] = np.mean(y)
    output["y_max"] = np.max(y)
    output["y_var"] = np.var(y)
    output["y_std"] = np.std(y)

    # Metrics
    output["Corr. Coef."] = np.corrcoef(t.to_list(), y.to_list())[0, 1]
    output["R2"] = sklearn.metrics.r2_score(t, y)
    output["MSE"] = sklearn.metrics.mean_squared_error(t, y)
    output["MAE"] = sklearn.metrics.mean_absolute_error(t, y)
    t2 = t[t > 0]
    y2 = y[t > 0]
    output["MAPE"] = np.mean(np.abs(t2 - y2) / np.abs(t2))
    output["Median of APE"] = np.median(np.abs(t2 - y2) / np.abs(t2))
    output["WAPE"] = np.sum(np.abs(t - y)) / np.sum(t)
    output["APE of total"] = np.abs(np.sum(t) - np.sum(y)) / np.sum(t)

    # Output dataframe
    df_output = pd.DataFrame(output.values(), index=output.keys(), columns =["score"])

    # Show scores
    if show_scores:
        pd.options.display.float_format = '{:.3f}'.format
        display(df_output)
        pd.options.display.float_format = None

    # yyplot
    if yyplot:
        plt.figure(figsize=(8,8))
        lim_min = np.min([np.min(t), np.min(y)])
        lim_max = np.max([np.max(t), np.max(y)])
        plt.xlabel("Prediction")
        plt.ylabel("Actual")
        plt.xlim(lim_min, lim_max)
        plt.ylim(lim_min, lim_max)
        plt.scatter(y, t, s=6)
        plt.plot([lim_min,lim_max], [lim_min,lim_max], linestyle=":", color="black")
        if yyplot_png:
            plt.savefig(yyplot_filename + ".png", dpi=300, bbox_inches='tight', pad_inches=0)
        if yyplot_svg:
            plt.savefig(yyplot_filename + ".svg", bbox_inches='tight', pad_inches=0)
        plt.show()

    return df_output

# Evaluate binary classification
def evaluate_binary_classification(t, y, threshold=0.5, roc_auc_curve=True, roc_auc_curve_png=True, roc_auc_curve_svg=False, show_scores=True, roc_auc_curve_filename="roc_auc_curve"):
    # Error check
    if len(t) != len(y):
        raise ValueError("len(t) != len(y) : " + str(len(t)) + " != " + str(len(y)))

    # Stats
    output = {}
    output["N"] = len(t)
    output["N_positive"] = np.count_nonzero(t)
    output["N_negative"] = len(t) - np.count_nonzero(t)

    # Metrics
    y_binary = (y >= threshold).astype(int)
    output["Accuracy"] = sklearn.metrics.accuracy_score(t, y_binary)
    output["Precision"] = sklearn.metrics.precision_score(t, y_binary)
    output["Recall"] = sklearn.metrics.recall_score(t, y_binary)
    output["F-value"] = sklearn.metrics.f1_score(t, y_binary)
    output["ROC-AUC"] = sklearn.metrics.roc_auc_score(t, y)

    # Output dataframe
    df_output = pd.DataFrame(output.values(), index=output.keys(), columns =["score"])

    # Show scores
    if show_scores:
        pd.options.display.float_format = '{:.3f}'.format
        display(df_output)
        pd.options.display.float_format = None

    # ROC-AUC Curve
    if roc_auc_curve:
        fpr, tpr, _ = sklearn.metrics.roc_curve(t, y)
        plt.figure(figsize=(8,8))
        plt.plot(fpr, tpr, linestyle="-"),
        plt.plot([0, 1], [0, 1], linestyle=":", color="black")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        if roc_auc_curve_png:
            plt.savefig(roc_auc_curve_filename + ".png", dpi=300, bbox_inches='tight', pad_inches=0)
        if roc_auc_curve_svg:
            plt.savefig(roc_auc_curve_filename + ".svg", bbox_inches='tight', pad_inches=0)
        plt.show()
    
    return df_output

# Query execution function
def run_query(project, query, df_mode=False, output_table=None, partition_column=None, query_output=True):
    if "CREATE_OR_REPLACE_TABLE" in query:
        query_source = query.replace("CREATE_OR_REPLACE_TABLE", "")
    else:
        query_source = query
    if query_output:
        p = Path("_last_query.sql")
        if p.exists():
            p.chmod(0o644)
        with open("_last_query.sql", "w") as f:
            f.write(query_source)
        p.chmod(0o444)
    bq_client = bigquery.Client(project)
    bqs_client = bigquery_storage.BigQueryReadClient()
    dry_run_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    query_job_dry_run = bq_client.query(query_source, job_config=dry_run_config)
    print("Query will process {:.3f} GB.".format(query_job_dry_run.total_bytes_processed / 1024 / 1024 / 1024))
    if output_table is not None:
        df_mode = False
        if partition_column is not None:
            additional_query = f"DROP TABLE IF EXISTS `{output_table}`;\n"
            additional_query += f"CREATE TABLE `{output_table}`\n"
            additional_query += f"PARTITION BY {partition_column}\n"
        else:
            additional_query = f"CREATE OR REPLACE TABLE `{output_table}`\n"
        if "CREATE_OR_REPLACE_TABLE" in query:
            query = query.replace("CREATE_OR_REPLACE_TABLE", additional_query + "AS\n")
        else:
            query = additional_query + "AS\n" + query
    start = time.time()
    if query_output:
        p = Path("_last_query.sql")
        if p.exists():
            p.chmod(0o644)
        with open("_last_query.sql", "w") as f:
            f.write(query)
        p.chmod(0o444)
    if df_mode:
        result = bq_client.query(query).to_dataframe(bqs_client)
    else:
        result = bq_client.query(query).result(timeout=3600*6)
    if output_table is not None:
        table = bq_client.get_table(output_table)
        table.description = query_source[:16000]
        bq_client.update_table(table, ["description"])
    print(str(time.time() - start) + "[sec]")
    return result

# Copy table
def copy_table(project, source_table, destination_table):
    bq_client = bigquery.Client(project)
    job_config = bigquery.CopyJobConfig()
    job_config.write_disposition = "WRITE_TRUNCATE"
    job = bq_client.copy_table(source_table, destination_table, job_config=job_config)
    job.result()
    table = bq_client.get_table(destination_table)
    table.description = f"Source table: {source_table}"
    bq_client.update_table(table, ["description"])

# Read schema
def read_schema(table):
    project_id = table.split(".", 1)[0]
    dataset = table.split(".")[1]
    table = table.split(".")[2]
    command = f"bq show --project_id={project_id} {dataset}.{table}"
    result = subprocess.run(command.split(" "), capture_output=True, text=True)
    print(result.stdout)
    display(HTML(f"<a href='https://console.cloud.google.com/bigquery?project={project_id}&ws=!1m5!1m4!4m3!1s{project_id}!2s{dataset}!3s{table}' style='color:red'>BQ Link</a>"))

# Get schema list
def get_table_info(table):
    project_id = table.split(".", 1)[0]
    dataset = table.split(".")[1]
    table = table.split(".")[2]
    bq_client = bigquery.Client(project_id)
    table_ref = bq_client.dataset(dataset).table(table)
    table = bq_client.get_table(table_ref)
    schema_list = []
    for field in table.schema:
        schema_list.append({"name": field.name, "type": field.field_type})
    return table, schema_list

# Close excel file
def close_excel_file(filename):
    script = f"""
        tell application "Microsoft Excel"
            set fileName to "{filename}"
            set fileFound to false
            set wbList to name of workbooks
            repeat with wbName in wbList
                if wbName as string is equal to fileName then
                    set fileFound to true
                    exit repeat
                end if
            end repeat
            
            if fileFound then
                tell workbook fileName to close saving no
            end if
        end tell
    """
    subprocess.run(["osascript", "-e", script])

# Dataframe to Google Spread Sheet
# pip install gspread gspread-dataframe google-auth-oauthlib google-authが必要。
# 以下の認証も必要
# gcloud auth application-default login --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/spreadsheets,https://www.googleapis.com/auth/drive
def dataframe_to_spreadsheet(df, prefix=""):
    # google-authライブラリで認証情報を取得
    creds, _ = google_auth_default()

    # 取得した認証情報をgspread.authorizeに渡す
    gc = gspread.authorize(creds)

    # 現在時刻の取得
    postfix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    # ファイル名生成
    if prefix != "":
        filename = prefix + "_" + postfix
    else:
        filename = postfix

    # スプレッドシートを開く
    spreadsheet = gc.create(filename)

    # ワークシートを選択
    worksheet = spreadsheet.get_worksheet(0) # 最初のシートを選択する場合

    # DataFrameをワークシートに書き出す (A1セルから書き込みが始まります)
    set_with_dataframe(worksheet, df)

    # アクセス用URLを返す
    return spreadsheet.url

# Google Spread Sheet to Dataframe
# pip install gspread gspread-dataframe google-auth-oauthlib google-authが必要。
# 以下の認証も必要
# gcloud auth application-default login --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/spreadsheets,https://www.googleapis.com/auth/drive
def spreadsheet_to_dataframe(sheet_key, sheet_name, sheet_range="", start_row=1, column_row=0):
    # google-authライブラリで認証情報を取得
    creds, _ = google_auth_default()

    # 取得した認証情報をgspread.authorizeに渡す
    gc = gspread.authorize(creds)

    # sheetの読み込み
    spreadsheet = gc.open_by_key(sheet_key)
    worksheet = spreadsheet.worksheet(sheet_name)

    # Dataの読み込み
    if sheet_range == "":
        data = worksheet.get_all_values()
    else:
        data = worksheet.get(sheet_range)
    df = pd.DataFrame(data[start_row:], columns=data[column_row])

    # Dataframeを返す
    return df