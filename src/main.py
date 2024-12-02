# from oldatasets.NuImages import NuImagesBEVDataset

# path = "/run/user/17937/gvfs/smb-share:server=gpfs-cluster,share=databases/GeneralDatabases/nuImages"
# dataset = NuImagesBEVDataset(dataroot=path, 
#             output_path='./data/NuImages/OpenLABEL', 
#             save_bevdataset=True)
# 

from TrainingLogger import TrainingLogger
import matplotlib.pyplot as plt

MODEL_OUTPUT_PATH = "./tmp/models"
MODEL_NAME = "model1"

def update(fig, axes):
    tl = TrainingLogger(model_out_path=MODEL_OUTPUT_PATH, model_name=MODEL_NAME)
    metrics = tl.get_epoch_metrics()
    
    axes[0].plot(metrics["train"]["loss"], c="r", label = "Train")
    axes[0].plot(metrics["eval"]["loss"], c="b", label = "Eval")

    axes[1].plot(metrics["train"]["mean_iou"], c="r", label = "Train")
    axes[1].plot(metrics["eval"]["mean_iou"], c="b", label = "Eval")
    
    plt.pause(0.1)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

def main():
    execution = True
    
    f, axs = plt.subplots(1, 2)
    axs[0].set_title("Loss"); axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss"); axs[0].legend()
    axs[1].set_title("Mean IOU"); axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("mean_iou"); axs[1].legend()

    plt.ion()

    while execution:        
        update(f, axs)

if __name__ == "__main__":
    main()