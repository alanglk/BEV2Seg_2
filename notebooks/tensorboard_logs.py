import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Ruta a la carpeta que contiene los runs
models_path     = "./models"
selected_type   = "bev"
map_selection   = { "bev": "segformer_bev", "nu": "segformer_nu_formatted" }
runs_path = os.path.join(models_path, map_selection[selected_type], "runs")


# SCALAR TAGS:
# "eval/loss", "eval/mean_accuracy", "eval/mean_iou", "eval/overall_accuracy", "eval/runtime", "eval/samples_per_second", "eval/steps_per_second"
# "train/epoch", "train/grad_norm", "train/learning_rate", "train/loss", "train/total_flos", "train/train_loss", "train/train_runtime", "train/train_samples_per_second", "train/train_steps_per_second"
scalar_tag = "eval/loss"


# Diccionario para guardar datos de cada modelo
models_data = {}

# Iterar sobre subcarpetas (cada subcarpeta es un modelo)
for model_name in os.listdir(runs_path):
    model_path = os.path.join(runs_path, model_name)
    if not os.path.isdir(model_path):
        continue

    # Buscar archivos de eventos en la subcarpeta
    for root, _, files in os.walk(model_path):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_file = os.path.join(root, file)
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()

                if scalar_tag in ea.Tags()["scalars"]:
                    scalar_events = ea.Scalars(scalar_tag)
                    steps = [e.step for e in scalar_events]
                    values = [e.value for e in scalar_events]
                    models_data[model_name] = (steps, values)

# Graficar con matplotlib
plt.figure(figsize=(10, 6))
for model_name, (steps, values) in models_data.items():
    plt.plot(steps, values, label=model_name)

plt.xlabel("Step")
plt.ylabel(scalar_tag.capitalize())
plt.title(f"{scalar_tag.capitalize()} over time for each model")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
