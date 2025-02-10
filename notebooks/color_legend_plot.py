from oldatasets.NuImages.nulabels import nulabels
import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(6, 4))
for i, label in enumerate(nulabels):
    print(f"{label.color}, {label.name}")
    col = tuple(c/255 for c in label.color)
    ax.add_patch(plt.Rectangle((0, i), 1, 1, color= col))
    ax.text(1.05, i + 0.5, label.name, va='center', ha='left', fontsize=12)

# Ajustar los límites y el aspecto del gráfico
ax.set_xlim(0, 2)
ax.set_ylim(0, len(nulabels))
ax.axis('off')  # Para no mostrar los ejes

# Mostrar el gráfico
plt.show()
