import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt

target_labels,predicted_labels = pickle.load(open('../out-data.pkl', 'rb'))


def classifaction_report_csv(report):
	print(report)
	title = 'Classification Report'
	report_data = []
	lines = [x for x in report.split('\n') if len(x) > 0]
	for i,line in enumerate(lines[1:]):
	    row = {}
	    row_data = [x for x in line.split("  ") if len(x) > 0]
	    row['class'] = row_data[0]
	    row['precision'] = float(row_data[1])
	    row['recall'] = float(row_data[2])
	    row['f1_score'] = float(row_data[3])
	    row['support'] = float(row_data[4])
	    report_data.append(row)
	dataframe = pd.DataFrame.from_dict(report_data)
	dataframe.to_csv(title+'.csv', index = False)





target_names = ['blade','bomb','knife']
report = classification_report(target_labels,predicted_labels,target_names=target_names)
classifaction_report_csv(report)


y_true = [target_names[x] for x in target_labels]
y_pred = [target_names[x] for x in predicted_labels]



cm_array = confusion_matrix(y_true,y_pred)
def plot_confusion_matrix(y_true,y_pred):
	true_labels = np.unique(y_true)
	pred_labels = np.unique(y_pred)
	# print(cm_array[:-1,:-1])
	plt.imshow(cm_array[:-1,:-1], cmap=plt.cm.Blues)
	plt.title("Confusion matrix", fontsize=16)
	# cbar = plt.colorbar(fraction=0.046, pad=0.04)
	cbar.set_label('Number of images')
	xtick_marks = np.arange(len(true_labels))
	ytick_marks = np.arange(len(pred_labels))
	plt.xticks(xtick_marks, true_labels, rotation=90)
	plt.yticks(ytick_marks,pred_labels)
	plt.ylabel('True label', fontsize=14)
	plt.xlabel('Predicted label', fontsize=14)
	fig_size = plt.rcParams["figure.figsize"]
	fig_size[0] = 12
	fig_size[1] = 12
	plt.rcParams["figure.figsize"] = fig_size
	plt.savefig('Confusion_Mat.jpg')

plot_confusion_matrix(y_true,y_pred)