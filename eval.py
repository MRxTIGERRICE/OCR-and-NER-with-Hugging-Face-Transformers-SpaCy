import os
import csv
import pandas as pd

def performance(TP, FP, FN):
    
    if (TP+FP) == 0:
        precision = "NaN"
    else:
        precision = TP/float((TP+FP))
        
    if (TP+FN) == 0:
        recall = "NaN"
    else:
        recall = TP/float((TP+FN))
    
    if (recall!="NaN") and (precision!="NaN"):
        f1_score = (2.0*precision*recall)/(precision+recall)
    else:
        f1_score = "NaN"
    
    return precision, recall, f1_score
    
def get_dataset_metrics(true_labels, pred_labels):
    
    metrics_dict = dict()
    
    for true_label, pred_label in zip(true_labels, pred_labels):
        if true_label not in metrics_dict:
            metrics_dict[true_label] = {"TP":0, "FP":0, "FN":0, "Support":0}
        
        if true_label != "OTHER":
            metrics_dict[true_label]["Support"] += 1
            
            if true_label == pred_label:
                metrics_dict[true_label]["TP"] += 1
            
            elif pred_label == "OTHER":
                metrics_dict[true_label]["FN"] += 1
            
        else:
            if pred_label != "OTHER":
                metrics_dict[pred_label]["FP"] += 1
           
    df = pd.DataFrame()
    
    df_list = []  # List to collect temporary DataFrames
    for field in metrics_dict:
        precision, recall, f1_score = performance(metrics_dict[field]["TP"], metrics_dict[field]["FP"], metrics_dict[field]["FN"])
        support = metrics_dict[field]["Support"]
        if field != "OTHER":  # Skip "OTHER" field
            temp_df = pd.DataFrame([[precision, recall, f1_score, support]],columns=["Precision", "Recall", "F1-Score", "Support"],index=[field])
            df_list.append(temp_df)  # Collect temp DataFrame into the list
# Concatenate all collected DataFrames into a single DataFrame
    # if df_list:
    df = pd.concat(df_list)
    return df

def get_doc_labels(doc_true, doc_pred):

    true_labels = [row[-1] for row in csv.reader(open(doc_true, "r"))]
    pred_labels = [row[-1] for row in csv.reader(open(doc_pred, "r"))]
    # print(pred_labels)
    return true_labels, pred_labels

def get_dataset_labels(true_path, pred_path, save=False):
    
    y_true, y_pred = [], []
    
    for true_file in os.listdir(true_path):
        for pred_file in os.listdir(pred_path):
            if (".tsv" in true_file) and (".tsv" in pred_file):
                if true_file == pred_file:
                    # print('getting here')
                    true_file, pred_file = f"{true_path}/{true_file}", f"{pred_path}/{pred_file}"
                    true_labels, pred_labels = get_doc_labels(true_file, pred_file)
                    
                    y_true.extend(true_labels)
                    y_pred.extend(pred_labels)
    # print(y_pred)
    df = get_dataset_metrics(y_true, y_pred)
    print(df)
    if save == True:
        df.to_csv("eval_metrics.tsv")



if __name__ == "__main__":
    doc_true = f"dataset/val_w_ann/boxes_transcripts_labels"
    doc_pred = f'dataset/val/boxes_transcripts'
    get_dataset_labels(doc_true, doc_pred, save=True)

        
        
        
    
    
    
