#!/bin/bash
version="val"
mkdir -p "./tmp/BEVDataset/$version/"  # Asegurar que la carpeta de destino exista

for file in ./tmp/NuImagesFormatted/$version/*_semantic.png; do
    if [ -f "$file" ]; then  # Verificar que el archivo existe
        filename=$(basename "$file")  # Obtener solo el nombre del archivo
        newname="${filename/_semantic.png/_raw_semantic.png}"  # Reemplazar sufijo
        cp "$file" "./tmp/BEVDataset/$version/$newname"  # Copiar con el nuevo nombre
        echo "Copiado: $file -> ./tmp/BEVDataset/$version/$newname"
    fi
done
