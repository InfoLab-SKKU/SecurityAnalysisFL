import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(history, title):
    print(f"{history.metrics_centralized = }")

    global_accuracy_centralised = history.metrics_centralized["accuracy"]
    #global_loss_centralised = history.metrics_centralized["loss"]
    round = [data[0] for data in global_accuracy_centralised]
    acc = [100.0 * data[1] for data in global_accuracy_centralised]
    #loss = [data[1] for data in global_loss_centralised]
    plt.plot(round, acc, color="blue", label="Accuracy")
    #plt.plot(round, loss, color="red", label="loss")
    plt.grid()
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Round")
    plt.title(title)
    plt.legend()
    # Show plot
    plt.show()


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def smooth_plot(data, title, path, smoothing_window=5):
    print(f"{data = }++++++++++")

    global_accuracy_distributed = data.metrics_distributed["accuracy"]
    global_accuracy_disributed_without_targeted = data.metrics_distributed["accuracy_excluding_poisoned"]
    global_recolt_centralised = data.metrics_distributed["recall"]
    global_asr_centralised = data.metrics_centralized["asr"]
    global_asr_distributed = data.metrics_distributed["asr"]
    #global_loss_centralised = data.loss_distributed["loss"]
    global_f1_centralised = data.metrics_distributed["f1_score"]
    global_precision_centralised = data.metrics_distributed["precision"]

    round = [data[0] for data in global_accuracy_distributed]
    #acc = [100.0 * data[1] for data in global_accuracy_centralised]
    acc = [data[1] for data in global_accuracy_distributed]
    acc_without_target = [data[1] for data in global_accuracy_disributed_without_targeted]
    recolt = [data[1] for data in global_recolt_centralised]
    #loss = [data[1] for data in global_loss_centralised]
    f1 = [data[1] for data in global_f1_centralised]
    precision = [data[1] for data in global_precision_centralised]
    asr = [data[1] for data in global_asr_centralised]


    # Apply smoothing
    if smoothing_window > 1:
        acc_smooth = moving_average(acc, smoothing_window)
        acc_without_target_smooth = moving_average(acc_without_target, smoothing_window)
        recolt_smooth = moving_average(recolt, smoothing_window)
        #loss_smooth = moving_average(loss, smoothing_window)
        f1_smooth = moving_average(f1, smoothing_window)
        round_smooth = round[:len(acc_smooth)]
        precision_smooth = moving_average(precision, smoothing_window)
        asr_smooth = moving_average(asr, smoothing_window)
    else:
        acc_smooth = acc
        acc_without_target_smooth = acc_without_target
        round_smooth = round
        recolt_smooth = recolt
        #loss_smooth = loss
        f1_smooth = f1
        precision_smooth = precision
        asr_smooth = asr

    # Save the smoothed results to a CSV file
    csv_path = Path(path) / 'results.csv'
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "accuracy", "accuracy_without_targeted", "recall", "precision", "f1_score", "asr"])
        writer.writerows(zip(round_smooth, acc_smooth, acc_without_target_smooth, recolt_smooth, precision_smooth, f1_smooth, asr))

    print(f"Results saved to {csv_path}")
    plt.plot(round_smooth, acc_smooth, color="blue", label="Accuracy")
    plt.grid()
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=1)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Round")
    #plt.title(title)
    plt.legend()
    plt.savefig(path / 'graph.png', bbox_inches='tight', )
    plt.savefig(path / 'graph.pdf', bbox_inches='tight')
    #plt.show()
    print("==========================================================", path)


# import numpy as np
# import matplotlib.pyplot as plt
#
# # Function to smooth the data using a moving average
# def smooth_curve(data, window_size=5):
#     return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
#
# # Loss values for the three models
# loss_values_model_1 = [ ]  # Insert the actual values here
# loss_values_model_2 = [...]  # Insert the actual values here
# loss_values_model_3 = [...]  # Insert the actual values here
#
# # Smoothing the loss curves
# smoothed_loss_model_1 = smooth_curve(loss_values_model_1)
# smoothed_loss_model_2 = smooth_curve(loss_values_model_2)
# smoothed_loss_model_3 = smooth_curve(loss_values_model_3)
#
# # Calculate differences between models
# diff_model_1_2 = np.array(smoothed_loss_model_1) - np.array(smoothed_loss_model_2[:len(smoothed_loss_model_1)])
# diff_model_1_3 = np.array(smoothed_loss_model_1) - np.array(smoothed_loss_model_3[:len(smoothed_loss_model_1)])
# diff_model_2_3 = np.array(smoothed_loss_model_2[:len(smoothed_loss_model_3)]) - np.array(smoothed_loss_model_3)
#
# # Plotting the smoothed loss curves and difference plots
# plt.figure(figsize=(12, 10))
#
# # Smoothed loss curves
# plt.subplot(2, 1, 1)
# plt.plot(smoothed_loss_model_1, label='Model 1', color='blue', linestyle='-', marker='o')
# plt.plot(smoothed_loss_model_2, label='Model 2', color='green', linestyle='-', marker='x')
# plt.plot(smoothed_loss_model_3, label='Model 3', color='red', linestyle='-', marker='s')
# plt.title('Smoothed Training Loss Comparison: Model 1 vs Model 2 vs Model 3')
# plt.xlabel('Training Round')
# plt.ylabel('Smoothed Loss')
# plt.legend()
# plt.grid(True)
#
# # Difference plot
# plt.subplot(2, 1, 2)
# plt.plot(diff_model_1_2, label='Model 1 - Model 2', color='purple', linestyle='-', marker='o')
# plt.plot(diff_model_1_3, label='Model 1 - Model 3', color='orange', linestyle='-', marker='x')
# plt.plot(diff_model_2_3, label='Model 2 - Model 3', color='brown', linestyle='-', marker='s')
# plt.title('Difference in Loss Between Models')
# plt.xlabel('Training Round')
# plt.ylabel('Loss Difference')
# plt.legend()
# plt.grid(True)
#
# # Display the plot
# plt.tight_layout()
# plt.show()
